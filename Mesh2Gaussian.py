"""
Mesh2Gaussian.py

【前提条件】
入力となる3Dメッシュファイル（.glb 推奨、.gltf, .obj等にも対応）は、
Blenderなどの3Dソフトウェア上で既存シーンの点群に合わせて
「位置・向き・スケールが配置・調整済み」であることを前提とします。

【処理概要】
1. GLBファイルに内包されている「本来のテクスチャ画像 (baseColorTexture)」と「UV座標」から
   正確なカラー（RGB）をサンプリングし、頂点座標を維持したままガウシアン化。
2. 配置済みの3DメッシュをColmapのカメラ座標からレンダリングし、物体の輪郭に完全一致した「高精度2Dシルエットマスク」を自動生成・保存。
3. 座標変換（回転・移動・スケーリング）を一切挟まず、GaussianModel.concat_gaussians で既存シーンに直接結合してPLY保存。
"""

import os
import sys
import argparse
import subprocess
from typing import Optional, List, Tuple
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

# ==============================================================================
# GaussianEditor / 3D Gaussian Splatting モジュールのインポート
# ==============================================================================
from gaussiansplatting.scene.gaussian_model import GaussianModel
from gaussiansplatting.scene.vanilla_gaussian_model import GaussianModel as VanillaGaussianModel
from gaussiansplatting.scene import Scene
from gaussiansplatting.arguments import ModelParams, PipelineParams
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.utils.graphics_utils import BasicPointCloud


# ==============================================================================
# 1. GLBテクスチャサンプリング & ガウシアン化 (テクスチャ色をそのまま反映)
# ==============================================================================
def sample_colors_from_submesh(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_indices: np.ndarray,
    default_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
) -> np.ndarray:
    """
    GLB/GLTFのPBRマテリアル (baseColorTexture, baseColorFactor) や UV座標から
    サンプリング点の色 (RGB [0, 1]) を正確に補間・取得する。
    """
    n_pts = points.shape[0]
    if n_pts == 0:
        return np.empty((0, 3), dtype=np.float32)

    # 1. テクスチャ画像およびベースカラー係数の抽出
    tex_img = None
    base_factor = None
    mat = getattr(mesh.visual, "material", None)

    if mat is not None:
        # GLB/GLTF PBRマテリアルの baseColorTexture
        if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
            tex_img = mat.baseColorTexture.convert("RGB")
        # 一般的なテクスチャ画像
        elif hasattr(mat, "image") and mat.image is not None:
            tex_img = mat.image.convert("RGB")

        # baseColorFactor (乗算カラー係数)
        if hasattr(mat, "baseColorFactor") and mat.baseColorFactor is not None:
            base_factor = np.asarray(mat.baseColorFactor, dtype=np.float32)[:3]
        elif hasattr(mat, "diffuse") and mat.diffuse is not None:
            bf = np.asarray(mat.diffuse, dtype=np.float32)[:3]
            base_factor = bf / 255.0 if np.max(bf) > 1.0 else bf

    # 2. テクスチャ画像 + UV座標が存在する場合: 重心座標で補間サンプリング
    if tex_img is not None and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
        try:
            tex_np = np.asarray(tex_img, dtype=np.float32) / 255.0
            tex_h, tex_w = tex_np.shape[:2]

            faces = mesh.faces[face_indices]
            tri_verts = mesh.vertices[faces]
            barycentric = trimesh.triangles.points_to_barycentric(tri_verts, points)

            # 面ごとのUVを重心座標で補間
            uvs = mesh.visual.uv[faces]
            sampled_uvs = np.sum(uvs * barycentric[:, :, None], axis=1)

            # UV座標からピクセルインデックスを算出 (リピート・反転対応)
            u = np.clip(sampled_uvs[:, 0] % 1.0, 0.0, 1.0) * (tex_w - 1)
            v = np.clip((1.0 - (sampled_uvs[:, 1] % 1.0)), 0.0, 1.0) * (tex_h - 1)
            u_idx = np.clip(np.round(u).astype(int), 0, tex_w - 1)
            v_idx = np.clip(np.round(v).astype(int), 0, tex_h - 1)
            colors = tex_np[v_idx, u_idx, :3]

            if base_factor is not None:
                colors = colors * base_factor

            return colors.astype(np.float32)
        except Exception as e:
            print(f"  [Warning] テクスチャサンプリング失敗: {e}. 代替カラーを使用します。")

    # 3. 頂点カラーが存在する場合のフォールバック
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
        v_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32)
        if np.max(v_colors) > 1.0:
            v_colors = v_colors / 255.0
        faces = mesh.faces[face_indices]
        tri_verts = mesh.vertices[faces]
        barycentric = trimesh.triangles.points_to_barycentric(tri_verts, points)
        colors = np.sum(v_colors[faces] * barycentric[:, :, None], axis=1)
        return colors.astype(np.float32)

    # 4. マテリアル単色またはデフォルト色
    if base_factor is not None:
        return np.full((n_pts, 3), base_factor, dtype=np.float32)
    return np.full((n_pts, 3), default_color, dtype=np.float32)


def sample_mesh_with_texture(mesh_path: str, num_points: int = 200000) -> Tuple[np.ndarray, np.ndarray]:
    """
    GLBメッシュを読み込み、テクスチャをそのまま反映した表面点群 (points, colors) を抽出する。
    """
    loaded = trimesh.load(mesh_path, process=False)

    submeshes = []
    if isinstance(loaded, trimesh.Scene):
        # GLBの各階層ノードのトランスフォーム行列を適用してフラット化
        dumped = loaded.dump()
        for m in dumped:
            if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0 and len(m.faces) > 0:
                submeshes.append(m)
    elif isinstance(loaded, trimesh.Trimesh):
        submeshes.append(loaded)
    else:
        raise ValueError(f"対応していないメッシュ形式です: {type(loaded)}")

    if len(submeshes) == 0:
        raise ValueError("有効なメッシュジオメトリが見つかりませんでした。")

    # 表面積に応じてサンプリング点数を按分
    areas = np.array([max(m.area, 1e-9) for m in submeshes], dtype=np.float64)
    weights = areas / np.sum(areas)
    sub_counts = np.round(weights * num_points).astype(int)

    diff = num_points - np.sum(sub_counts)
    if diff != 0:
        sub_counts[np.argmax(areas)] += diff

    all_points = []
    all_colors = []

    for mesh, count in zip(submeshes, sub_counts):
        if count <= 0:
            continue
        pts, face_indices = trimesh.sample.sample_surface(mesh, count)
        cols = sample_colors_from_submesh(mesh, pts, face_indices)
        all_points.append(pts)
        all_colors.append(cols)

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32)
    return points, colors


def convert_mesh_to_gaussian(
    mesh_path: str,
    output_gs_path: str,
    num_points: int = 200000,
    camera_extent: float = 1.0,
    refine_prompt: Optional[str] = None,
    use_refine: bool = False,
) -> VanillaGaussianModel:
    """
    GLBメッシュ本来のテクスチャ画像とワールド座標を完全に維持したままガウシアン化する。
    """
    print(f"\n[Step 1] GLBメッシュのガウシアン化を開始: {mesh_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_gs_path)), exist_ok=True)

    if use_refine and refine_prompt:
        print(f"  -> train_from_mesh.py を呼び出してプロンプト誘導リファインを実行: '{refine_prompt}'")
        cmd = [
            sys.executable,
            "train_from_mesh.py",
            "--mesh",
            mesh_path,
            "--save_path",
            output_gs_path,
            "--prompt",
            refine_prompt,
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"train_from_mesh.py の実行中にエラーが発生しました (exit code: {result.returncode})")
        
        obj_gaussian = VanillaGaussianModel(sh_degree=3)
        obj_gaussian.load_ply(output_gs_path)
        return obj_gaussian
    else:
        print(f"  -> 内包テクスチャ画像 (baseColorTexture) から色を直接サンプリング (目標点数: {num_points:,})")
        points, colors = sample_mesh_with_texture(mesh_path, num_points)
        
        # 点群の作成（メッシュ本来の3Dワールド座標とテクスチャRGBカラーを使用）
        pcd = BasicPointCloud(points=points, colors=colors, normals=None)
        
        obj_gaussian = VanillaGaussianModel(sh_degree=3)
        obj_gaussian.create_from_pcd(pcd, spatial_lr_scale=camera_extent)
        
        # 不透明度をしっかり可視化される値に設定
        obj_gaussian._opacity.data = torch.ones_like(obj_gaussian._opacity.data) * 4.595  # sigmoid(4.595) ~ 0.99
        
        obj_gaussian.save_ply(output_gs_path)
        print(f"  -> テクスチャ付きガウシアン化完了: {output_gs_path} (点数: {obj_gaussian.get_xyz.shape[0]:,})")
        return obj_gaussian


# ==============================================================================
# 2. 既存シーンとColmapカメラのロード
# ==============================================================================
def load_base_scene(
    source_path: str,
    model_path: Optional[str] = None,
    iteration: int = -1,
    sh_degree: int = 3,
) -> Tuple[GaussianModel, Scene]:
    """
    既存の3DGSシーンおよびColmapカメラパラメータを読み込む。
    """
    print(f"\n[Step 2] 既存の3DGSシーンおよびカメラデータの読み込み: {source_path}")
    if model_path is None:
        model_path = source_path

    model_params = ModelParams(parser=argparse.ArgumentParser(), sent_args=[])
    model_params.source_path = source_path
    model_params.model_path = model_path
    model_params.sh_degree = sh_degree
    model_params.white_background = False
    model_params.eval = False

    base_gaussian = GaussianModel(sh_degree=sh_degree)
    
    scene = Scene(
        args=model_params,
        gaussians=base_gaussian,
        load_iteration=iteration,
        shuffle=False
    )
    print(f"  -> シーン読み込み成功 (既存点数: {base_gaussian.get_xyz.shape[0]:,}, カメラ数: {len(scene.getTrainCameras())})")
    return base_gaussian, scene


# ==============================================================================
# 3. Colmapカメラからの3Dメッシュレンダリング & 2Dシルエットマスク自動生成
# ==============================================================================
def render_mesh_silhouette_mask(
    obj_gaussian: VanillaGaussianModel,
    cam,
    pipe: PipelineParams,
    threshold: float = 0.01,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    配置済みの3Dメッシュ（ガウシアン）をColmapカメラ視点から直接レンダリングし、
    メッシュが写っているピクセルを抽出した高精度2Dシルエットマスクを生成する。
    """
    bg_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
    render_pkg = render(cam, obj_gaussian, pipe, bg_black)
    
    rendered_image = render_pkg["render"]  # (3, H, W)
    pixel_intensity = torch.max(rendered_image, dim=0)[0]  # (H, W)
    mask_tensor = pixel_intensity > threshold

    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_np, mode="L")
    return mask_tensor, mask_image


def generate_multiview_masks(
    obj_gaussian: VanillaGaussianModel,
    cameras: List,
    output_dir: str,
    pipe: PipelineParams,
):
    """
    すべてのColmapカメラ視点からメッシュをレンダリングし、物体が写っているカメラのマスクを一括出力する。
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Mask Generation] 全Colmapカメラからの高精度2Dシルエットマスク一括生成を開始...")

    saved_count = 0
    for idx, cam in enumerate(cameras):
        mask_tensor, mask_image = render_mesh_silhouette_mask(obj_gaussian, cam, pipe)
        if mask_tensor.sum() > 0:
            mask_filename = os.path.join(output_dir, f"mask_cam_{idx:04d}.png")
            mask_image.save(mask_filename)
            saved_count += 1
    print(f"  -> メッシュが写っている {saved_count} 視点の2Dシルエットマスクを保存しました: {output_dir}")


# ==============================================================================
# 4. シーンの結合と保存 (直接マージ)
# ==============================================================================
def merge_and_save_scene(
    base_gaussian: GaussianModel,
    new_object_gaussian: VanillaGaussianModel,
    output_merged_ply: str,
):
    """
    GaussianModel.concat_gaussians を使用してシーンを結合し、PLYとして保存する。
    """
    print(f"\n[Step 3] シーンの結合と保存処理...")
    before_count = base_gaussian.get_xyz.shape[0]
    add_count = new_object_gaussian.get_xyz.shape[0]

    # メッシュ座標の移動・回転を行わず、そのまま直接結合
    base_gaussian.concat_gaussians(new_object_gaussian)
    after_count = base_gaussian.get_xyz.shape[0]

    print(f"  -> 結合完了: 既存 {before_count:,} 点 + 追加 {add_count:,} 点 = 合計 {after_count:,} 点")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_merged_ply)), exist_ok=True)
    base_gaussian.save_ply(output_merged_ply)
    print(f"[Success] 統合された 3DGS PLY ファイルを保存しました: {output_merged_ply}\n")


# ==============================================================================
# メイン実行エントリポイント
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="【配置済みGLBメッシュ用】GLBのテクスチャをそのまま反映してガウシアン化し、Colmapカメラからの2Dマスク生成とシーン統合を行うプログラム"
    )
    # 必須・基本入力 (GLB推奨)
    parser.add_argument("--mesh", "-m", type=str, required=True, help="Blender等で配置済みの3Dメッシュファイル (.glb 推奨, .gltf, .obj等)")
    parser.add_argument("--source_path", "-s", type=str, required=True, help="Colmapデータセットのパス (sparse/0/ があるディレクトリ)")
    parser.add_argument("--model_path", type=str, default=None, help="学習済み3DGSモデルのパス (指定しない場合は source_path)")
    parser.add_argument("--iteration", type=int, default=-1, help="読み込む学習イテレーション番号 (デフォルト: 最新)")
    parser.add_argument("--output_ply", "-o", type=str, default="merged.ply", help="結合後の出力PLYファイルパス")

    # メッシュ変換設定
    parser.add_argument("--num_points", "-n", type=int, default=200000, help="メッシュからサンプリングするガウシアン点数 (デフォルト: 200,000)")
    parser.add_argument("--refine_prompt", "-p", type=str, default=None, help="train_from_mesh によるリファイン用プロンプト (任意)")
    parser.add_argument("--use_refine", action="store_true", help="train_from_mesh.py による InstructPix2Pix リファインを実行する")

    # 2Dシルエットマスク出力設定
    parser.add_argument("--cam_id", type=int, default=0, help="2Dマスクをレンダリング保存する基準カメラのID (デフォルト: 0)")
    parser.add_argument("--save_mask", type=str, default="silhouette_mask.png", help="レンダリングした高精度2Dシルエットマスクの保存先パス")
    parser.add_argument("--multiview_mask_dir", type=str, default=None, help="メッシュが写っている全カメラ視点から2Dマスクを一括出力するディレクトリ (任意)")

    args = parser.parse_args()
    pipe = PipelineParams(parser=argparse.ArgumentParser(), sent_args=[])

    # 1. GLBメッシュのガウシアン化 (テクスチャ色・ワールド座標を完全保持)
    temp_gs_path = "temp_mesh_gs.ply"
    obj_gaussian = convert_mesh_to_gaussian(
        mesh_path=args.mesh,
        output_gs_path=temp_gs_path,
        num_points=args.num_points,
        refine_prompt=args.refine_prompt,
        use_refine=args.use_refine,
    )

    # 2. 既存のベースシーンおよびカメラの読み込み
    base_gaussian, scene = load_base_scene(
        source_path=args.source_path,
        model_path=args.model_path,
        iteration=args.iteration,
    )

    train_cams = scene.getTrainCameras()
    cam = train_cams[min(args.cam_id, len(train_cams) - 1)]

    # 3. 配置済みメッシュから正確な2Dシルエットマスクを自動抽出・保存
    if args.save_mask:
        _, mask_img = render_mesh_silhouette_mask(obj_gaussian, cam, pipe)
        os.makedirs(os.path.dirname(os.path.abspath(args.save_mask)), exist_ok=True)
        mask_img.save(args.save_mask)
        print(f"\n[Mask] カメラ {args.cam_id} から見た高精度2Dシルエットマスクを保存しました: {args.save_mask}")

    # （任意）全視点からのマルチビュー2Dマスク一括出力
    if args.multiview_mask_dir:
        generate_multiview_masks(
            obj_gaussian=obj_gaussian,
            cameras=train_cams,
            output_dir=args.multiview_mask_dir,
            pipe=pipe,
        )

    # 4. シーンの結合と出力 (位置合わせ処理は一切挟まず、直接マージ)
    merge_and_save_scene(
        base_gaussian=base_gaussian,
        new_object_gaussian=obj_gaussian,
        output_merged_ply=args.output_ply,
    )

    # 一時ファイルの削除
    if os.path.exists(temp_gs_path):
        os.remove(temp_gs_path)


if __name__ == "__main__":
    main()
