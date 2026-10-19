"""
Mesh2Gaussian.py

【前提条件】
入力となる3Dメッシュファイル（.glb 推奨、.gltf, .obj等にも対応）は、
Blenderなどの3Dソフトウェア上で既存シーンの点群に合わせて
「位置・向き・スケールが配置・調整済み」であることを前提とします。

【処理概要】
1. GLBファイルに内包されている「本来のテクスチャ画像 (baseColorTexture)」と「UV座標」から
   正確なカラー（RGB）をサンプリングし、頂点座標を維持したままガウシアン化。
2. 【内部Zバッファ合成方式によるZ-fighting解消 & オクルージョン処理】
   3D座標は1ミリも動かさず、PyTorch上で内部的なZバッファ（深度バッファテスト）を管理。
   同一平面上の重なりポリゴンでは「最大インデックス番号のメッシュ」を優先して手前として判定。
   テクスチャなしメッシュが手前にある場合は、奥のテクスチャを黒で上書きして2Dマスクから完全遮蔽・除外。
3. 座標変換（回転・移動・スケーリング）を一切挟まず、GaussianModel.concat_gaussians で既存シーンに直接結合してPLY保存。
"""

import os
import sys
import re
import argparse
import subprocess
from typing import Optional, List, Tuple, Dict
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
# 0. テクスチャ有無・メッシュインデックス判定ユーティリティ
# ==============================================================================
def extract_mesh_index(name_or_str: Optional[str], default_idx: int) -> int:
    """
    メッシュ名（例: 'mesh_05', 'Layer_12', 'Decal-3'）から数値を抽出する。
    数値が見つからない場合は出現順 (default_idx) を返す。
    """
    if name_or_str:
        matches = re.findall(r'\d+', str(name_or_str))
        if matches:
            return int(matches[-1])  # 末尾の数値をインデックスとする
    return default_idx


def split_submeshes_by_material(mesh: trimesh.Trimesh, base_idx: int) -> List[Tuple[trimesh.Trimesh, int]]:
    """
    1つのメッシュ内に複数マテリアル (face_materials) が存在する場合、
    マテリアルごとにサブメッシュに分割し、それぞれにインデックスを付与する。
    """
    if not hasattr(mesh, "visual") or mesh.visual is None:
        return [(mesh, base_idx)]

    face_materials = getattr(mesh.visual, "face_materials", None)
    if face_materials is None or len(face_materials) != len(mesh.faces):
        return [(mesh, base_idx)]

    try:
        unique_mats = np.unique(face_materials)
        if len(unique_mats) <= 1:
            return [(mesh, base_idx)]

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
                sub_list.append((sub_mesh, base_idx))

        if len(sub_list) > 0:
            return sub_list
    except Exception:
        pass

    return [(mesh, base_idx)]


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


def sample_mesh_with_texture_and_priority(
    mesh_path: str,
    num_points: int = 200000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    GLBメッシュを読み込み、各サブメッシュのインデックス番号を抽出。
    表面点群 (points)、色 (colors)、テクスチャ有無 (has_tex)、優先度インデックス (priorities) を返す。
    """
    loaded = trimesh.load(mesh_path, process=False)

    indexed_submeshes = []
    if isinstance(loaded, trimesh.Scene):
        for default_idx, (name, geom) in enumerate(loaded.geometry.items()):
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
                mesh_idx = extract_mesh_index(name, default_idx)
                transform = loaded.graph.get(name)[0] if name in loaded.graph else np.eye(4)
                geom_transformed = geom.copy()
                geom_transformed.apply_transform(transform)
                indexed_submeshes.append((geom_transformed, mesh_idx, name))
    elif isinstance(loaded, trimesh.Trimesh):
        indexed_submeshes.append((loaded, 0, "root"))
    else:
        raise ValueError(f"対応していないメッシュ形式です: {type(loaded)}")

    if len(indexed_submeshes) == 0:
        raise ValueError("有効なメッシュジオメトリが見つかりませんでした。")

    final_submeshes = []
    for m, m_idx, name in indexed_submeshes:
        split_results = split_submeshes_by_material(m, m_idx)
        for s_mesh, s_idx in split_results:
            final_submeshes.append((s_mesh, s_idx, name))

    areas = np.array([max(m[0].area, 1e-9) for m in final_submeshes], dtype=np.float64)
    weights = areas / np.sum(areas)
    sub_counts = np.round(weights * num_points).astype(int)

    diff = num_points - np.sum(sub_counts)
    if diff != 0:
        sub_counts[np.argmax(areas)] += diff

    all_points = []
    all_colors = []
    all_has_tex = []
    all_priorities = []

    tex_submesh_cnt = 0
    print(f"  -> メッシュモデルのインデックス解析:")
    for (mesh, p_idx, name), count in zip(final_submeshes, sub_counts):
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
        all_priorities.append(np.full(len(pts), p_idx, dtype=np.int32))

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32)
    has_tex_array = np.concatenate(all_has_tex, axis=0).astype(bool)
    priorities = np.concatenate(all_priorities, axis=0).astype(np.int32)

    unique_priorities = np.unique(priorities)
    print(f"  -> 検出されたメッシュインデックス一覧: {unique_priorities.tolist()} (最大値: {int(np.max(unique_priorities))} が最前面)")
    print(f"  -> サブメッシュ構成: 全 {len(final_submeshes)} 個中、テクスチャあり {tex_submesh_cnt} 個 / なし {len(final_submeshes) - tex_submesh_cnt} 個")

    return points, colors, has_tex_array, priorities


def convert_mesh_to_gaussian(
    mesh_path: str,
    output_gs_path: str,
    num_points: int = 200000,
    camera_extent: float = 1.0,
) -> Tuple[VanillaGaussianModel, Dict[int, Tuple[VanillaGaussianModel, bool]]]:
    """
    GLBメッシュをガウシアン化し、以下を返す:
    1. full_gaussian: シーン結合用の全体モデル (ワールド座標をそのまま完全保持)
    2. sub_gaussians_by_idx: 内部Zバッファテスト用の {インデックス番号: (サブガウシアン, テクスチャ有無)} 辞書
    """
    print(f"\n[Step 1] GLBメッシュのガウシアン化を開始: {mesh_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_gs_path)), exist_ok=True)

    points, colors, has_tex_array, priorities = sample_mesh_with_texture_and_priority(mesh_path, num_points)

    # 全体モデルの初期化 (座標は一切変換しない)
    pcd_full = BasicPointCloud(points=points, colors=colors, normals=None)
    full_gaussian = VanillaGaussianModel(sh_degree=3)
    full_gaussian.create_from_pcd(pcd_full, spatial_lr_scale=camera_extent)
    full_gaussian._opacity.data = torch.ones_like(full_gaussian._opacity.data) * 4.595  # ~0.99
    full_gaussian.save_ply(output_gs_path)

    # 内部Zバッファテスト用: インデックスごとにサブガウシアンモデルをグループ化
    sub_gaussians_by_idx = {}
    unique_indices = np.unique(priorities)
    
    for idx in unique_indices:
        idx_mask = (priorities == idx)
        pts_sub = points[idx_mask]
        cols_sub = colors[idx_mask]
        # このインデックスのメッシュがテクスチャを持つか
        idx_has_tex = bool(np.any(has_tex_array[idx_mask]))

        pcd_sub = BasicPointCloud(points=pts_sub, colors=cols_sub, normals=None)
        sub_gs = VanillaGaussianModel(sh_degree=0)
        sub_gs.create_from_pcd(pcd_sub, spatial_lr_scale=camera_extent)
        sub_gs._opacity.data = torch.ones_like(sub_gs._opacity.data) * 4.595
        sub_gaussians_by_idx[idx] = (sub_gs, idx_has_tex)

    print(f"  -> ガウシアン化完了: {output_gs_path} (インデックスレイヤー数: {len(sub_gaussians_by_idx)})")
    return full_gaussian, sub_gaussians_by_idx


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
# 3. 【内部Zバッファ合成方式】による2Dシルエットマスク自動生成
#    (3D座標を動かさず、ピクセル単位の深度テストとインデックス優先度で合成)
# ==============================================================================
def render_mesh_silhouette_mask_with_zbuffer(
    sub_gaussians_by_idx: Dict[int, Tuple[VanillaGaussianModel, bool]],
    cam,
    pipe: PipelineParams,
    depth_tolerance: float = 0.005,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    【内部Zバッファ合成アルゴリズム】
    3D座標は1ミリも変更せず、PyTorch上でピクセル単位のZバッファテストを実行:
    1. 各メッシュのレンダリング深度マップ (depth_3dgs) を比較。
    2. 深度が手前にあるメッシュがZバッファを更新。
    3. 同一平面（|D1 - D2| <= depth_tolerance: Z-fighting領域）では、インデックス番号が大きいメッシュが優先上書き。
    4. 手前になったメッシュがテクスチャありならマスクを「白(True)」、なし(オクルーダー)なら「黒(False)」で上書き。
    """
    H, W = cam.image_height, cam.image_width
    bg_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    # 内部Zバッファ (初期値: 無限遠) と 2Dマスクバッファ (初期値: 全て黒/False)
    z_buffer = torch.full((H, W), float("inf"), dtype=torch.float32, device="cuda")
    mask_buffer = torch.zeros((H, W), dtype=torch.bool, device="cuda")

    # インデックス昇順 (0 -> 1 -> ... -> 最大番号) でレンダリング・Zテスト
    sorted_indices = sorted(sub_gaussians_by_idx.keys())

    for idx in sorted_indices:
        sub_gs, has_tex = sub_gaussians_by_idx[idx]

        # 3D座標は元のままレンダリング
        render_pkg = render(
            viewpoint_camera=cam,
            pc=sub_gs,
            pipe=pipe,
            bg_color=bg_black,
        )
        
        rendered_image = render_pkg["render"]        # (3, H, W)
        rendered_depth = render_pkg["depth_3dgs"][0] # (H, W)

        # メッシュが描画されている有効ピクセル
        valid_pixel = (rendered_image.max(dim=0)[0] > 0.01) & (rendered_depth > 0.001)

        # Zテスト条件:
        # A. 明確に手前にある場合: rendered_depth < (z_buffer - depth_tolerance)
        # B. 同一平面 (Z-fighting 領域): |rendered_depth - z_buffer| <= depth_tolerance
        #    (昇順で処理しているため、同等深度ならインデックスが大きい方が手前として上書き)
        is_closer = rendered_depth < (z_buffer - depth_tolerance)
        is_coplanar_higher = torch.abs(rendered_depth - z_buffer) <= depth_tolerance

        update_pixels = valid_pixel & (is_closer | is_coplanar_higher)

        # Zバッファの更新
        z_buffer[update_pixels] = rendered_depth[update_pixels]

        # マスクバッファの更新 (ペインターズ＆オクルージョン)
        if has_tex:
            # テクスチャありメッシュが手前: 白 (True) で上書き
            mask_buffer[update_pixels] = True
        else:
            # テクスチャなしメッシュ (オクルーダー) が手前: 黒 (False) で上書き遮蔽！
            mask_buffer[update_pixels] = False

    # 最終的な 2D 白黒画像を出力
    mask_np = (mask_buffer.cpu().numpy() * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_np, mode="L")
    return mask_buffer, mask_image


def generate_multiview_masks(
    sub_gaussians_by_idx: Dict[int, Tuple[VanillaGaussianModel, bool]],
    cameras: List,
    output_dir: str,
    pipe: PipelineParams,
    depth_tolerance: float = 0.005,
):
    """
    すべてのColmapカメラ視点から、内部Zバッファ合成方式による2Dマスクを一括出力する。
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[Mask Generation] 内部Zバッファ合成による2Dシルエットマスク一括生成を開始...")

    saved_count = 0
    for idx, cam in enumerate(cameras):
        mask_tensor, mask_image = render_mesh_silhouette_mask_with_zbuffer(
            sub_gaussians_by_idx=sub_gaussians_by_idx,
            cam=cam,
            pipe=pipe,
            depth_tolerance=depth_tolerance,
        )
        if mask_tensor.sum() > 0:
            mask_filename = os.path.join(output_dir, f"mask_cam_{idx:04d}.png")
            mask_image.save(mask_filename)
            saved_count += 1
    print(f"  -> 正確な2Dシルエットマスクを {saved_count} 視点保存しました: {output_dir}")


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

    # 3D座標は一切動かさず、そのまま直接マージ
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
        description="【配置済みGLB用】内部Zバッファ合成方式により3D座標を動かさずZ-fighting解消＆オクルージョン処理を行うプログラム"
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
    parser.add_argument("--multiview_mask_dir", type=str, default=None, help="全カメラ視点から2Dマスクを一括出力するディレクトリ (任意)")
    parser.add_argument("--depth_tolerance", type=float, default=0.005, help="同一平面判定の深度トレランス (Z-fighting解消用, デフォルト: 0.005)")

    args = parser.parse_args()
    pipe = PipelineParams(parser=argparse.ArgumentParser(), sent_args=[])

    # 1. GLBメッシュのガウシアン化 & インデックス・テクスチャ識別
    temp_gs_path = "temp_mesh_gs.ply"
    full_gaussian, sub_gaussians_by_idx = convert_mesh_to_gaussian(
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

    # 3. ★【内部Zバッファ合成方式】による2Dシルエットマスクの自動抽出・保存
    # (3D座標は1ミリも動かさず、深度比較とインデックス優先度で手前判定＆遮蔽)
    if args.save_mask:
        _, mask_img = render_mesh_silhouette_mask_with_zbuffer(
            sub_gaussians_by_idx=sub_gaussians_by_idx,
            cam=cam,
            pipe=pipe,
            depth_tolerance=args.depth_tolerance,
        )
        os.makedirs(os.path.dirname(os.path.abspath(args.save_mask)), exist_ok=True)
        mask_img.save(args.save_mask)
        print(f"\n[Mask] カメラ {args.cam_id} から見た内部Zバッファ合成済み2Dマスクを保存しました: {args.save_mask}")

    # （任意）全視点からのマルチビュー2Dマスク一括出力
    if args.multiview_mask_dir:
        generate_multiview_masks(
            sub_gaussians_by_idx=sub_gaussians_by_idx,
            cameras=train_cams,
            output_dir=args.multiview_mask_dir,
            pipe=pipe,
            depth_tolerance=args.depth_tolerance,
        )

    # 4. シーンの結合と出力 (元座標のまま完全にマージ)
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
