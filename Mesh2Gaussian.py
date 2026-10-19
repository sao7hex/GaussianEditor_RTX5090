"""
Mesh2Gaussian.py

【前提条件】
入力となる3Dメッシュファイル（.glb 推奨、.gltf, .obj等にも対応）は、
Blenderなどの3Dソフトウェア上で既存シーンの点群に合わせて
「位置・向き・スケールが配置・調整済み」であることを前提とします。

【処理概要】
1. GLBファイルに内包されている「本来のテクスチャ画像 (baseColorTexture)」と「UV座標」から
   正確なカラー（RGB）をサンプリングし、頂点座標を維持したままガウシアン化。
2. 【オクルージョン（遮蔽）考慮型2Dマスク生成機能】
   テクスチャが存在しないメッシュを「オクルーダー（遮蔽物）」として機能させ、
   『テクスチャがないメッシュに隠れたテクスチャ領域』を2Dマスクから完全に除外してレンダリング。
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
# 0. テクスチャ有無の判定ユーティリティ
# ==============================================================================
def split_submeshes_by_material(mesh: trimesh.Trimesh) -> List[trimesh.Trimesh]:
    """
    1つのメッシュ内に複数マテリアル (face_materials) が存在する場合、
    マテリアルごとにサブメッシュに分割する。
    """
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return [mesh]

    face_materials = getattr(mesh.visual, "face_materials", None)
    if face_materials is None or len(face_materials) != len(mesh.faces):
        return [mesh]

    try:
        unique_mats = np.unique(face_materials)
        if len(unique_mats) <= 1:
            return [mesh]

        sub_list = []
        materials = getattr(mesh.visual.material, "materials", None)

        for mat_idx in unique_mats:
            face_mask = (face_materials == mat_idx)
            face_indices = np.where(face_mask)[0]
            if len(face_indices) == 0:
                continue

            sub = mesh.submesh([face_indices], append=False)
            sub_mesh = sub[0] if isinstance(sub, list) and len(sub) > 0 else sub

            if materials is not None and 0 <= mat_idx < len(materials):
                sub_mesh.visual.material = materials[mat_idx]

            if len(sub_mesh.vertices) > 0 and len(sub_mesh.faces) > 0:
                sub_list.append(sub_mesh)

        if len(sub_list) > 0:
            return sub_list
    except Exception:
        pass

    return [mesh]


def mesh_has_texture(mesh: trimesh.Trimesh) -> bool:
    """
    サブメッシュが有効なUV座標および画像テクスチャを持っているかを厳密に判定する。
    """
    has_uv = hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0
    if not has_uv:
        return False

    mat = getattr(mesh.visual, "material", None)
    if mat is None:
        return False

    if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
        return True
    if hasattr(mat, "image") and mat.image is not None:
        return True
    if hasattr(mat, "materials") and mat.materials is not None:
        for m in mat.materials:
            if hasattr(m, "baseColorTexture") and m.baseColorTexture is not None:
                return True
            if hasattr(m, "image") and m.image is not None:
                return True

    return False


# ==============================================================================
# 1. GLBテクスチャサンプリング & ガウシアン化
# ==============================================================================
def sample_colors_from_submesh(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_indices: np.ndarray,
    default_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    サブメッシュから色 (RGB [0, 1]) を取得し、同時にテクスチャから正常に取得できたかのマスク (has_tex_mask) を返す。
    """
    n_pts = points.shape[0]
    if n_pts == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=bool)

    tex_img = None
    base_factor = None
    mat = getattr(mesh.visual, "material", None)

    if mat is not None:
        if hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
            tex_img = mat.baseColorTexture.convert("RGBA")
        elif hasattr(mat, "image") and mat.image is not None:
            tex_img = mat.image.convert("RGBA")

        if hasattr(mat, "baseColorFactor") and mat.baseColorFactor is not None:
            base_factor = np.asarray(mat.baseColorFactor, dtype=np.float32)[:3]
        elif hasattr(mat, "diffuse") and mat.diffuse is not None:
            bf = np.asarray(mat.diffuse, dtype=np.float32)[:3]
            base_factor = bf / 255.0 if np.max(bf) > 1.0 else bf

    if tex_img is not None and hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
        try:
            tex_np = np.asarray(tex_img, dtype=np.float32) / 255.0
            tex_h, tex_w = tex_np.shape[:2]

            faces = mesh.faces[face_indices]
            tri_verts = mesh.vertices[faces]
            barycentric = trimesh.triangles.points_to_barycentric(tri_verts, points)

            uvs = mesh.visual.uv[faces]
            sampled_uvs = np.sum(uvs * barycentric[:, :, None], axis=1)

            valid_uv = np.all(np.isfinite(sampled_uvs), axis=1)
            is_zero_uv = np.all(np.abs(uvs) < 1e-5, axis=(1, 2))

            u = np.clip(sampled_uvs[:, 0] % 1.0, 0.0, 1.0) * (tex_w - 1)
            v = np.clip((1.0 - (sampled_uvs[:, 1] % 1.0)), 0.0, 1.0) * (tex_h - 1)
            u_idx = np.clip(np.round(u).astype(int), 0, tex_w - 1)
            v_idx = np.clip(np.round(v).astype(int), 0, tex_h - 1)
            colors = tex_np[v_idx, u_idx, :3]

            if base_factor is not None:
                colors = colors * base_factor

            has_tex_mask = valid_uv & (~is_zero_uv)
            if tex_np.shape[2] >= 4:
                alphas = tex_np[v_idx, u_idx, 3]
                has_tex_mask = has_tex_mask & (alphas > 0.05)

            return colors.astype(np.float32), has_tex_mask
        except Exception as e:
            print(f"  [Warning] テクスチャサンプリング失敗: {e}")

    has_tex_mask = np.zeros(n_pts, dtype=bool)

    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
        v_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32)
        if np.max(v_colors) > 1.0:
            v_colors = v_colors / 255.0
        faces = mesh.faces[face_indices]
        tri_verts = mesh.vertices[faces]
        barycentric = trimesh.triangles.points_to_barycentric(tri_verts, points)
        colors = np.sum(v_colors[faces] * barycentric[:, :, None], axis=1)
        return colors.astype(np.float32), has_tex_mask

    if base_factor is not None:
        return np.full((n_pts, 3), base_factor, dtype=np.float32), has_tex_mask
    return np.full((n_pts, 3), default_color, dtype=np.float32), has_tex_mask


def sample_mesh_with_texture(
    mesh_path: str,
    num_points: int = 200000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GLBメッシュを読み込み、テクスチャを反映した点群と、
    各点がテクスチャを持つかどうかの真偽値配列 (has_tex_array) を返す。
    """
    loaded = trimesh.load(mesh_path, process=False)

    raw_submeshes = []
    if isinstance(loaded, trimesh.Scene):
        dumped = loaded.dump()
        for m in dumped:
            if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0 and len(m.faces) > 0:
                raw_submeshes.append(m)
    elif isinstance(loaded, trimesh.Trimesh):
        raw_submeshes.append(loaded)
    else:
        raise ValueError(f"対応していないメッシュ形式です: {type(loaded)}")

    if len(raw_submeshes) == 0:
        raise ValueError("有効なメッシュジオメトリが見つかりませんでした。")

    submeshes = []
    for m in raw_submeshes:
        submeshes.extend(split_submeshes_by_material(m))

    areas = np.array([max(m.area, 1e-9) for m in submeshes], dtype=np.float64)
    weights = areas / np.sum(areas)
    sub_counts = np.round(weights * num_points).astype(int)

    diff = num_points - np.sum(sub_counts)
    if diff != 0:
        sub_counts[np.argmax(areas)] += diff

    all_points = []
    all_colors = []
    all_has_tex = []

    tex_submesh_cnt = 0
    for mesh, count in zip(submeshes, sub_counts):
        if count <= 0:
            continue
        
        has_tex = mesh_has_texture(mesh)
        if has_tex:
            tex_submesh_cnt += 1

        pts, face_indices = trimesh.sample.sample_surface(mesh, count)
        cols, pt_tex_mask = sample_colors_from_submesh(mesh, pts, face_indices)
        
        final_pt_mask = pt_tex_mask if has_tex else np.zeros(len(pts), dtype=bool)

        all_points.append(pts)
        all_colors.append(cols)
        all_has_tex.append(final_pt_mask)

    print(f"  -> サブメッシュ構成: 全 {len(submeshes)} 個中、テクスチャあり {tex_submesh_cnt} 個 / なし {len(submeshes) - tex_submesh_cnt} 個")

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32)
    has_tex_array = np.concatenate(all_has_tex, axis=0).astype(bool)

    tex_points_cnt = np.sum(has_tex_array)
    print(f"  -> サンプリング点内訳: 全 {len(points):,} 点中、テクスチャ保持 {tex_points_cnt:,} 点 ({(tex_points_cnt/len(points))*100:.1f}%)")

    return points, colors, has_tex_array


def convert_mesh_to_gaussian(
    mesh_path: str,
    output_gs_path: str,
    num_points: int = 200000,
    camera_extent: float = 1.0,
) -> Tuple[VanillaGaussianModel, torch.Tensor]:
    """
    GLBメッシュをガウシアン化し、
    メッシュ全体のガウシアンと、各点がテクスチャを持つかのマスクテンソル (has_tex_mask) を返す。
    """
    print(f"\n[Step 1] GLBメッシュのガウシアン化を開始: {mesh_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_gs_path)), exist_ok=True)

    points, colors, has_tex_array = sample_mesh_with_texture(mesh_path, num_points)

    pcd_full = BasicPointCloud(points=points, colors=colors, normals=None)
    full_gaussian = VanillaGaussianModel(sh_degree=3)
    full_gaussian.create_from_pcd(pcd_full, spatial_lr_scale=camera_extent)
    
    # 遮蔽物（オクルーダー）としても完全に機能するよう不透明度を高く設定
    full_gaussian._opacity.data = torch.ones_like(full_gaussian._opacity.data) * 4.595  # ~0.99
    full_gaussian.save_ply(output_gs_path)

    has_tex_mask = torch.from_numpy(has_tex_array).to(device="cuda", dtype=torch.bool)
    print(f"  -> ガウシアン化完了: {output_gs_path}")
    return full_gaussian, has_tex_mask


# ==============================================================================
# 2. 既存シーンとColmapカメラのロード
# ==============================================================================
def load_base_scene(
    source_path: str,
    model_path: Optional[str] = None,
    iteration: int = -1,
    sh_degree: int = 3,
) -> Tuple[GaussianModel, Scene]:
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
# 3. オクルージョン考慮型 2Dシルエットマスク自動生成
#    (テクスチャなしメッシュをオクルーダーにして隠蔽部をマスクから除外)
# ==============================================================================
def render_mesh_silhouette_mask_with_occlusion(
    full_obj_gaussian: VanillaGaussianModel,
    has_tex_mask: torch.Tensor,
    cam,
    pipe: PipelineParams,
    threshold: float = 0.5,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    テクスチャなしメッシュを「オクルーダー（黒）」、テクスチャありメッシュを「白」として同時にレンダリング。
    手前のテクスチャなしメッシュによって隠蔽された奥のテクスチャ領域は黒くなり、マスクから完全に除外される。
    """
    num_pts = full_obj_gaussian.get_xyz.shape[0]

    # カラーオーバーライドの作成:
    # ・テクスチャあり点: [1.0, 1.0, 1.0] (白)
    # ・テクスチャなし点 (オクルーダー): [0.0, 0.0, 0.0] (黒)
    override_color = torch.zeros((num_pts, 3), dtype=torch.float32, device="cuda")
    override_color[has_tex_mask] = 1.0

    # 背景色は完全な黒 [0, 0, 0]
    bg_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    # GaussianEditor の override_color 機能を使用してレンダリング
    render_pkg = render(
        viewpoint_camera=cam,
        pc=full_obj_gaussian,
        pipe=pipe,
        bg_color=bg_black,
        override_color=override_color,
    )
    
    rendered_image = render_pkg["render"]  # (3, H, W)
    
    # 手前のオクルーダーに遮られず、白く視認できるピクセルのみを True (白) と判定
    # (オクルーダーに隠された奥のテクスチャは黒で遮蔽されるため threshold 以下になる)
    mask_tensor = rendered_image[0] > threshold

    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_np, mode="L")
    return mask_tensor, mask_image


def generate_multiview_masks(
    full_obj_gaussian: VanillaGaussianModel,
    has_tex_mask: torch.Tensor,
    cameras: List,
    output_dir: str,
    pipe: PipelineParams,
):
    """
    すべてのColmapカメラ視点から、オクルージョンを考慮した2Dシルエットマスクを一括出力する。
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Mask Generation] オクルージョン考慮型 2Dシルエットマスク一括生成を開始...")

    saved_count = 0
    for idx, cam in enumerate(cameras):
        mask_tensor, mask_image = render_mesh_silhouette_mask_with_occlusion(
            full_obj_gaussian=full_obj_gaussian,
            has_tex_mask=has_tex_mask,
            cam=cam,
            pipe=pipe,
        )
        if mask_tensor.sum() > 0:
            mask_filename = os.path.join(output_dir, f"mask_cam_{idx:04d}.png")
            mask_image.save(mask_filename)
            saved_count += 1
    print(f"  -> テクスチャが視認できる {saved_count} 視点の2Dシルエットマスクを保存しました: {output_dir}")


# ==============================================================================
# 4. シーンの結合と保存
# ==============================================================================
def merge_and_save_scene(
    base_gaussian: GaussianModel,
    obj_gaussian: VanillaGaussianModel,
    output_merged_ply: str,
):
    print(f"\n[Step 4] シーンの結合と保存処理...")
    before_count = base_gaussian.get_xyz.shape[0]
    add_count = obj_gaussian.get_xyz.shape[0]

    base_gaussian.concat_gaussians(obj_gaussian)
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
        description="【配置済みGLBメッシュ用】テクスチャなしメッシュをオクルーダーにして遮蔽領域をマスクから除外するプログラム"
    )
    # 必須・基本入力
    parser.add_argument("--mesh", "-m", type=str, required=True, help="Blender等で配置済みの3Dメッシュファイル (.glb 推奨, .gltf, .obj等)")
    parser.add_argument("--source_path", "-s", type=str, required=True, help="Colmapデータセットのパス (sparse/0/ があるディレクトリ)")
    parser.add_argument("--model_path", type=str, default=None, help="学習済み3DGSモデルのパス (指定しない場合は source_path)")
    parser.add_argument("--iteration", type=int, default=-1, help="読み込む学習イテレーション番号 (デフォルト: 最新)")
    parser.add_argument("--output_ply", "-o", type=str, default="merged.ply", help="結合後の出力PLYファイルパス")

    # メッシュ変換設定
    parser.add_argument("--num_points", "-n", type=int, default=200000, help="メッシュからサンプリングするガウシアン点数 (デフォルト: 200,000)")

    # 2Dシルエットマスク出力設定
    parser.add_argument("--cam_id", type=int, default=0, help="2Dマスクをレンダリング保存する基準カメラのID (デフォルト: 0)")
    parser.add_argument("--save_mask", type=str, default="silhouette_mask.png", help="レンダリングした高精度2Dシルエットマスクの保存先パス")
    parser.add_argument("--multiview_mask_dir", type=str, default=None, help="全カメラ視点からオクルージョン考慮型2Dマスクを一括出力するディレクトリ (任意)")

    args = parser.parse_args()
    pipe = PipelineParams(parser=argparse.ArgumentParser(), sent_args=[])

    # 1. GLBメッシュのガウシアン化 & テクスチャ識別
    temp_gs_path = "temp_mesh_gs.ply"
    full_gaussian, has_tex_mask = convert_mesh_to_gaussian(
        mesh_path=args.mesh,
        output_gs_path=temp_gs_path,
        num_points=args.num_points,
    )

    # 2. 既存のベースシーンおよびカメラの読み込み
    base_gaussian, scene = load_base_scene(
        source_path=args.source_path,
        model_path=args.model_path,
        iteration=args.iteration,
    )

    train_cams = scene.getTrainCameras()
    cam = train_cams[min(args.cam_id, len(train_cams) - 1)]

    # 3. ★オクルージョン考慮型 2Dシルエットマスクの自動抽出・保存
    # (テクスチャなしメッシュに隠された奥のテクスチャは自動的に黒くなり除外される)
    if args.save_mask:
        _, mask_img = render_mesh_silhouette_mask_with_occlusion(
            full_obj_gaussian=full_gaussian,
            has_tex_mask=has_tex_mask,
            cam=cam,
            pipe=pipe,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.save_mask)), exist_ok=True)
        mask_img.save(args.save_mask)
        print(f"\n[Mask] カメラ {args.cam_id} から見たオクルージョン考慮型2Dマスクを保存しました: {args.save_mask}")

    # （任意）全視点からのマルチビュー2Dマスク一括出力
    if args.multiview_mask_dir:
        generate_multiview_masks(
            full_obj_gaussian=full_gaussian,
            has_tex_mask=has_tex_mask,
            cameras=train_cams,
            output_dir=args.multiview_mask_dir,
            pipe=pipe,
        )

    # 4. シーンの結合と出力
    merge_and_save_scene(
        base_gaussian=base_gaussian,
        obj_gaussian=full_gaussian,
        output_merged_ply=args.output_ply,
    )

    # 一時ファイルの削除
    if os.path.exists(temp_gs_path):
        os.remove(temp_gs_path)


if __name__ == "__main__":
    main()
