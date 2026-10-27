"""
Mesh2Gaussian.py

【前提条件】
入力となる3Dメッシュファイル（.glb 推奨、.gltf, .obj等にも対応）は、
Blenderなどの3Dソフトウェア上で既存シーンの点群に合わせて
「位置・向き・スケールが配置・調整済み」であることを前提とします。

【処理概要】
1. GLBファイルに内包されている「本来のテクスチャ画像 (baseColorTexture)」と「UV座標」から
   正確なカラー（RGB）をサンプリングし、頂点座標を維持したままガウシアン化。
2. 【オクルーダー遮蔽のインテリジェント制御】
   ・「テクスチャが存在しないメッシュモデル（遮蔽物パーツ）」:
       オクルーダーとして機能し、手前にある場合は奥のテクスチャをマスクから遮蔽・除外。
   ・「テクスチャが存在するメッシュモデル内のテクスチャがない領域（デカールの余白・透明部分など）」:
       オクルーダー遮断は一切行わず、完全透過（スルー）して奥のレイヤーを描画。
3. 【内部Zバッファ合成方式によるZ-fighting解消】
   3D座標は1ミリも動かさず、PyTorch上で内部的なZバッファ（深度バッファテスト）を管理。
   同一平面上の重なりポリゴンでは「最大インデックス番号のメッシュ」を優先して手前として判定。
4. 座標系自動補正: COLMAP座標系 (Y-Down, Z-Forward) と glTF規格 (Y-Up, -Z Forward) の不一致を解消するため、デフォルトでX軸180度回転 (--rotate_x 180.0) を自動適用して既存シーンと完全整合。GaussianModel.concat_gaussians で直接結合してPLY保存。
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
    サブメッシュから色 (RGB [0, 1]) を取得し、同時に点単位でテクスチャが存在するかのマスク (has_tex_mask) を返す。
    UV未展開面や透明部分は has_tex_mask = False となる。
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



def apply_euler_rotation(points: np.ndarray, rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """
    点群に対してオイラー角 (度) による3軸回転を適用する。
    rx_deg=180.0, ry_deg=0.0, rz_deg=0.0 (COLMAP/glTF座標軸補正) の場合は厳密な符号反転 (Y -> -Y, Z -> -Z) を行う。
    """
    if rx_deg == 0.0 and ry_deg == 0.0 and rz_deg == 0.0:
        return points

    # X軸180度のみの場合は誤差のない直接符号反転 (Y -> -Y, Z -> -Z)
    if rx_deg == 180.0 and ry_deg == 0.0 and rz_deg == 0.0:
        pts = points.copy()
        pts[:, 1] = -pts[:, 1]
        pts[:, 2] = -pts[:, 2]
        return pts

    rad_x = np.radians(rx_deg)
    rad_y = np.radians(ry_deg)
    rad_z = np.radians(rz_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(rad_x), -np.sin(rad_x)],
        [0, np.sin(rad_x), np.cos(rad_x)]
    ], dtype=np.float32)

    Ry = np.array([
        [np.cos(rad_y), 0, np.sin(rad_y)],
        [0, 1, 0],
        [-np.sin(rad_y), 0, np.cos(rad_y)]
    ], dtype=np.float32)

    Rz = np.array([
        [np.cos(rad_z), -np.sin(rad_z), 0],
        [np.sin(rad_z), np.cos(rad_z), 0],
        [0, 0, 1]
    ], dtype=np.float32)

    R = Rz @ Ry @ Rx
    return (points @ R.T).astype(np.float32)


def sample_mesh_with_texture_and_priority(
    mesh_path: str,
    num_points: int = 200000,
    rotate_x: float = 0.0,
    rotate_y: float = 0.0,
    rotate_z: float = 0.0,
    mesh_log_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GLBメッシュを読み込み、一番最後のメッシュを除外した上でサンプリングを行う。
    points, colors, priorities を返す。
    デフォルトでは回転補正なし (rotate_x=0.0)。必要に応じてオイラー角を適用可能。
    mesh_log_path が指定されている場合、メッシュ一覧とサンプリング結果をログファイルに保存する。
    """
    loaded = trimesh.load(mesh_path, process=False)

    log_lines = [
        "=" * 70,
        "【Mesh2Gaussian メッシュ解析・サンプリングログ】",
        f"・入力メッシュ: {os.path.abspath(mesh_path)}",
        f"・サンプリング目標点数: {num_points:,} 点",
        f"・回転補正: X={rotate_x}°, Y={rotate_y}°, Z={rotate_z}°",
        "=" * 70,
    ]

    def log(msg: str):
        print(msg)
        log_lines.append(msg)

    indexed_submeshes = []
    if isinstance(loaded, trimesh.Scene):
        parents_map = getattr(loaded.graph.transforms, "parents", {})
        base_frame = loaded.graph.base_frame

        # シーングラフのノードを走査し、親ノードからの累積変換行列を正確に計算
        nodes_with_mesh = []
        for default_idx, node_name in enumerate(loaded.graph.nodes):
            geom_name = loaded.graph.transforms.node_data.get(node_name, {}).get("geometry", None)
            if geom_name is None and node_name in loaded.geometry:
                geom_name = node_name
            if geom_name is not None and geom_name in loaded.geometry:
                nodes_with_mesh.append((default_idx, node_name, geom_name))

        # node_data から取得できなかった場合は geometry.items() をフォールバック利用
        if len(nodes_with_mesh) == 0:
            for default_idx, (name, _) in enumerate(loaded.geometry.items()):
                nodes_with_mesh.append((default_idx, name, name))

        # 空でない有効なメッシュジオメトリを抽出
        valid_mesh_nodes = []
        for default_idx, node_name, geom_name in nodes_with_mesh:
            geom = loaded.geometry[geom_name]
            if isinstance(geom, trimesh.Trimesh) and len(geom.vertices) > 0 and len(geom.faces) > 0:
                valid_mesh_nodes.append((default_idx, node_name, geom_name))
            else:
                log(f"     [スキップ] 空のジオメトリ: ノード='{node_name}'")

        total_valid = len(valid_mesh_nodes)
        log(f"  -> GLB内の全メッシュ検出・解析 (有効メッシュ数: {total_valid} 個):")

        for mesh_count, (default_idx, node_name, geom_name) in enumerate(valid_mesh_nodes):
            geom = loaded.geometry[geom_name]
            mesh_idx = extract_mesh_index(node_name, default_idx)

            # ★ GLB内で一番最後（末尾 = mesh_count == total_valid - 1）に出現するメッシュを除外！
            if total_valid > 1 and mesh_count == total_valid - 1:
                log(f"     [除外] ★ 最後のメッシュ（#{mesh_count}）をスキップ: ノード='{node_name}', ジオメトリ='{geom_name}' (頂点数: {len(geom.vertices):,})")
                continue

            log(f"     [採用] メッシュ #{mesh_count}: ノード='{node_name}', ジオメトリ='{geom_name}' (Index: {mesh_idx}, 頂点数: {len(geom.vertices):,})")

            # 親ノードの取得と階層行列の合成計算 (T_parent_world @ T_child_local)
            parent_node = parents_map.get(node_name, base_frame)
            try:
                # 親ノードのワールド変換行列
                T_parent_world, _ = loaded.graph.get(frame_to=parent_node, frame_from=base_frame)
                # 親に対する子ノードのローカル変換行列
                T_child_local, _ = loaded.graph.get(frame_to=node_name, frame_from=parent_node)
                # 親ノードの行列と子のローカル行列を行列積で合成
                transform = T_parent_world @ T_child_local
            except Exception:
                # 万が一のフォールバック (単一ノード直接取得)
                transform = loaded.graph.get(node_name)[0] if node_name in loaded.graph else np.eye(4)

            geom_transformed = geom.copy()
            geom_transformed.apply_transform(transform)
            indexed_submeshes.append((geom_transformed, mesh_idx, node_name))
    elif isinstance(loaded, trimesh.Trimesh):
        # 単一メッシュしかない場合は警告しつつ保持
        log("  -> [Warning] 単一のTrimeshオブジェクトが読み込まれたため、末尾除外をスキップして処理します。")
        indexed_submeshes.append((loaded, 0, "root"))
    else:
        raise ValueError(f"対応していないメッシュ形式です: {type(loaded)}")

    if len(indexed_submeshes) == 0:
        raise ValueError("有効なメッシュジオメトリが見つかりませんでした（末尾メッシュ除外後の残りが0件です）。")

    final_submeshes = []
    for m, m_idx, name in indexed_submeshes:
        split_results = split_submeshes_by_material(m, m_idx)
        for s_mesh, s_idx in split_results:
            final_submeshes.append((s_mesh, s_idx, name))

    # 各メッシュの面積比率計算と最小サンプリング点数の保証 (極小メッシュでも消えないようにする)
    areas = np.array([max(m[0].area, 1e-9) for m in final_submeshes], dtype=np.float64)
    weights = areas / np.sum(areas)

    min_pts_guarantee = min(1000, max(100, num_points // (len(final_submeshes) * 5)))
    raw_counts = np.round(weights * (num_points - min_pts_guarantee * len(final_submeshes))).astype(int)
    sub_counts = np.maximum(raw_counts + min_pts_guarantee, min_pts_guarantee)

    diff = num_points - np.sum(sub_counts)
    if diff != 0:
        sub_counts[np.argmax(areas)] += diff

    all_points = []
    all_colors = []
    all_priorities = []

    log(f"\n  -> 各メッシュへのサンプリング点数割り当て結果 (合計: {num_points:,} 点):")
    for (mesh, p_idx, name), count in zip(final_submeshes, sub_counts):
        if count <= 0:
            log(f"     [スキップ] 点数0: ノード='{name}'")
            continue

        log(f"     ・ノード='{name}' (優先度Index: {p_idx}, 表面積: {mesh.area:.5f}): {count:,} 点をサンプリング")

        pts, face_indices = trimesh.sample.sample_surface(mesh, count)
        cols, has_tex_mask = sample_colors_from_submesh(mesh, pts, face_indices)

        # テクスチャが存在しない点（テクスチャなしメッシュ、UV未展開面、透過領域など）にはガウシアンを生成しない
        if not np.any(has_tex_mask):
            log(f"     [スキップ] テクスチャが存在しないためガウシアン生成を除外: ノード='{name}'")
            continue

        valid_pts = pts[has_tex_mask]
        valid_cols = cols[has_tex_mask]
        valid_priorities = np.full(len(valid_pts), p_idx, dtype=np.int32)

        if len(valid_pts) < count:
            log(f"     [テクスチャフィルタ] 非テクスチャ/透明領域 {count - len(valid_pts):,} 点を除外 -> 有効な {len(valid_pts):,} 点のみガウシアン化")

        all_points.append(valid_pts)
        all_colors.append(valid_cols)
        all_priorities.append(valid_priorities)

    if len(all_points) == 0:
        raise ValueError("テクスチャを持つ有効な点群が見つかりませんでした（すべての点にテクスチャが存在しないか透明です）。")

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32)
    priorities = np.concatenate(all_priorities, axis=0).astype(np.int32)

    # 座標系補正 (COLMAP系 OpenCV: Y-Down, Z-Forward vs glTF OpenGL: Y-Up, -Z Forward)
    if rotate_x != 0.0 or rotate_y != 0.0 or rotate_z != 0.0:
        points = apply_euler_rotation(points, rotate_x, rotate_y, rotate_z)
        log(f"  -> [座標系補正] メッシュ点群に回転補正を適用: X={rotate_x}°, Y={rotate_y}°, Z={rotate_z}°")

    unique_priorities = np.unique(priorities)
    log(f"  -> 検出されたメッシュインデックス一覧: {unique_priorities.tolist()} (最大値: {int(np.max(unique_priorities))} が最前面)")
    log(f"  -> 点単位サンプリング内訳: 全 {len(points):,} 点を生成")

    # ログファイルの保存 (output/logs/mesh_log.txt)
    if mesh_log_path:
        os.makedirs(os.path.dirname(os.path.abspath(mesh_log_path)), exist_ok=True)
        with open(mesh_log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"  -> [ログ保存] メッシュ解析ログを保存しました: {mesh_log_path}")

    return points, colors, priorities


def convert_mesh_to_gaussian(
    mesh_path: str,
    output_gs_path: str,
    num_points: int = 200000,
    camera_extent: float = 1.0,
    rotate_x: float = 0.0,
    rotate_y: float = 0.0,
    rotate_z: float = 0.0,
    mesh_log_path: Optional[str] = None,
) -> Tuple[VanillaGaussianModel, Dict[int, List[Tuple[VanillaGaussianModel, bool]]], Tuple[np.ndarray, np.ndarray]]:
    """
    GLBメッシュ（一番最後のメッシュ除外済み）をガウシアン化し、以下を返す:
    1. full_gaussian: シーン結合用の全体モデル (ワールド座標をそのまま完全保持)
    2. render_layers_by_idx: 内部Zバッファ用の {インデックス番号: [(サブガウシアン, is_occluder=False)]} 辞書
    3. mesh_bounds: (min_xyz, max_xyz) のバウンディングボックス
    """
    print(f"\n[Step 1] GLBメッシュのガウシアン化を開始: {mesh_path} (一番最後のメッシュを除外)")
    os.makedirs(os.path.dirname(os.path.abspath(output_gs_path)), exist_ok=True)

    # ログ出力先が指定されていない場合は output_gs_path と同階層の logs/mesh_log.txt に設定
    if mesh_log_path is None:
        mesh_log_path = os.path.join(os.path.dirname(os.path.abspath(output_gs_path)), "logs", "mesh_log.txt")

    points, colors, priorities = sample_mesh_with_texture_and_priority(
        mesh_path=mesh_path,
        num_points=num_points,
        rotate_x=rotate_x,
        rotate_y=rotate_y,
        rotate_z=rotate_z,
        mesh_log_path=mesh_log_path,
    )


    mesh_bounds = (points.min(axis=0), points.max(axis=0))

    # 全体モデルの初期化 (座標は一切変換しない)
    pcd_full = BasicPointCloud(points=points, colors=colors, normals=None)
    full_gaussian = VanillaGaussianModel(sh_degree=3)
    full_gaussian.create_from_pcd(pcd_full, spatial_lr_scale=camera_extent)
    full_gaussian._opacity.data = torch.ones_like(full_gaussian._opacity.data) * 4.595  # ~0.99
    full_gaussian.save_ply(output_gs_path)

    # 内部Zバッファテスト用レイヤーの構築 (オクルーダーは廃止し、すべてマスク描画対象 is_occluder=False とする)
    render_layers_by_idx: Dict[int, List[Tuple[VanillaGaussianModel, bool]]] = {}
    unique_indices = np.unique(priorities)
    
    for idx in unique_indices:
        idx_mask = (priorities == idx)
        render_layers_by_idx[idx] = []

        pts_layer = points[idx_mask]
        cols_layer = colors[idx_mask]
        pcd_layer = BasicPointCloud(points=pts_layer, colors=cols_layer, normals=None)
        gs_layer = VanillaGaussianModel(sh_degree=0)
        gs_layer.create_from_pcd(pcd_layer, spatial_lr_scale=camera_extent)
        gs_layer._opacity.data = torch.ones_like(gs_layer._opacity.data) * 4.595
        render_layers_by_idx[idx].append((gs_layer, False))  # is_occluder = False

    print(f"  -> ガウシアン化完了: {output_gs_path} (インデックスレイヤー数: {len(render_layers_by_idx)})")
    return full_gaussian, render_layers_by_idx, mesh_bounds



# ==============================================================================
# 2. 既存シーンとColmapカメラのロード
# ==============================================================================
def resolve_model_path(model_dir: str) -> Optional[str]:
    """
    model_dir 配下の構造を柔軟に探索し、3DGSモデルのパスまたはPLYファイルを返す:
    1. model_dir/point_cloud がある場合 -> model_dir
    2. model_dir 直下またはサブフォルダ内に point_cloud/ がある場合 -> その親
    3. model_dir 内に .ply ファイルがある場合 -> その .ply ファイルパス
    """
    if not os.path.exists(model_dir):
        return None

    # 1. model_dir/point_cloud が直接存在する場合
    if os.path.exists(os.path.join(model_dir, "point_cloud")):
        return model_dir

    # 2. サブディレクトリ内に point_cloud があるか探索
    for item in os.listdir(model_dir):
        sub = os.path.join(model_dir, item)
        if os.path.isdir(sub):
            if os.path.exists(os.path.join(sub, "point_cloud")):
                return sub

    # 3. 直下または配下に .ply ファイルが存在するか探索
    ply_files = []
    for root, _, files in os.walk(model_dir):
        for f in sorted(files):
            if f.lower().endswith(".ply") and "temp" not in f.lower():
                ply_files.append(os.path.join(root, f))
    if ply_files:
        # point_cloud.ply を優先
        for pf in ply_files:
            if "point_cloud.ply" in pf.lower():
                return pf
        return ply_files[-1]

    return None


def load_base_scene(
    source_path: str,
    model_path: Optional[str] = None,
    iteration: int = -1,
    sh_degree: int = 3,
) -> Tuple[GaussianModel, Scene]:
    print(f"\n[Step 2] 既存の3DGSシーンおよびカメラデータの読み込み:")
    print(f"  ・Colmapソースパス : {source_path}")
    print(f"  ・モデルパス       : {model_path}")

    direct_ply_file = None
    if model_path is not None:
        if os.path.isfile(model_path) and model_path.lower().endswith(".ply"):
            direct_ply_file = model_path
            model_params_path = os.path.dirname(os.path.abspath(model_path))
        else:
            model_params_path = model_path
    else:
        model_params_path = source_path

    dummy_parser = argparse.ArgumentParser()
    lp = ModelParams(dummy_parser)
    model_params = lp.extract(dummy_parser.parse_args([]))
    model_params.source_path = os.path.abspath(source_path)
    model_params.model_path = os.path.abspath(model_params_path)
    model_params.sh_degree = sh_degree
    model_params.white_background = False
    model_params.eval = False
    model_params.images = "images"
    model_params.resolution = -1
    model_params.data_device = "cuda"

    base_gaussian = GaussianModel(
        sh_degree=sh_degree,
        anchor_weight_init_g0=1.0,
        anchor_weight_init=0.1,
        anchor_weight_multiplier=2.0,
    )
    # direct_ply_file がある場合は Scene による自動PLY探索を抑制
    scene_load_iter = None if direct_ply_file else iteration

    scene = Scene(
        args=model_params,
        gaussians=base_gaussian,
        load_iteration=scene_load_iter,
        shuffle=False
    )

    # 直接PLYが指定されている場合は上書きロード
    if direct_ply_file and os.path.exists(direct_ply_file):
        print(f"  -> 指定されたPLYファイルを直接ロードします: {direct_ply_file}")
        base_gaussian.load_ply(direct_ply_file)
    elif scene.loaded_iter is None:
        # Scene がイテレーション番号を見つけられなかった場合、model_params_path 配下のPLYを救済探索
        resolved_ply = resolve_model_path(model_params_path)
        if resolved_ply and os.path.isfile(resolved_ply):
            print(f"  -> 学習済みPLYファイルを救済ロードします: {resolved_ply}")
            base_gaussian.load_ply(resolved_ply)
        else:
            print(f"  [Notice] 学習済みモデル (point_cloud/) が見つからなかったため、Colmapの初期疎点群を使用します。")

    # ロード後の点数に合わせて補助バッファ（max_radii2D等）を安全に初期化・同期
    curr_n_pts = base_gaussian.get_xyz.shape[0]
    if hasattr(base_gaussian, "max_radii2D"):
        base_gaussian.max_radii2D = torch.zeros(curr_n_pts, device="cuda")
    if hasattr(base_gaussian, "_generation"):
        base_gaussian._generation = torch.zeros(curr_n_pts, dtype=torch.int64, device="cuda")

    print(f"  -> シーン読み込み成功 (既存点数: {curr_n_pts:,}, カメラ数: {len(scene.getTrainCameras())})")
    return base_gaussian, scene


# ==============================================================================
# 3. 【内部Zバッファ合成方式】による2Dシルエットマスク自動生成
#    (テクスチャなしメッシュのみオクルーダー化 / テクスチャありメッシュの余白は完全透過)
# ==============================================================================
def render_mesh_silhouette_mask_with_zbuffer(
    render_layers_by_idx: Dict[int, List[Tuple[VanillaGaussianModel, bool]]],
    cam,
    pipe: PipelineParams,
    depth_tolerance: float = 0.005,
) -> Tuple[torch.Tensor, Image.Image]:
    """
    【内部Zバッファ合成アルゴリズム】
    ・3D座標は完全固定。
    ・インデックス昇順でレンダリング＆Zバッファ更新。
    ・「テクスチャなしメッシュモデル」: 手前の場合は mask_buffer = False (黒) で奥を遮蔽。
    ・「テクスチャありメッシュモデルのテクスチャ領域」: 手前の場合は mask_buffer = True (白) で上書き。
    ・「テクスチャありメッシュモデル内のテクスチャなし余白」: オクルーダー遮断を行わず完全透過（奥のテクスチャをそのまま維持）。
    """
    H, W = cam.image_height, cam.image_width
    bg_black = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    z_buffer = torch.full((H, W), float("inf"), dtype=torch.float32, device="cuda")
    mask_buffer = torch.zeros((H, W), dtype=torch.bool, device="cuda")

    sorted_indices = sorted(render_layers_by_idx.keys())

    for idx in sorted_indices:
        layers = render_layers_by_idx[idx]

        for sub_gs, is_occluder in layers:
            render_pkg = render(
                viewpoint_camera=cam,
                pc=sub_gs,
                pipe=pipe,
                bg_color=bg_black,
            )
            
            rendered_image = render_pkg["render"]
            rendered_depth = render_pkg["depth_3dgs"][0]

            valid_pixel = (rendered_image.max(dim=0)[0] > 0.01) & (rendered_depth > 0.001)

            # Zテスト (手前 または 同一平面でのインデックス優先)
            is_closer = rendered_depth < (z_buffer - depth_tolerance)
            is_coplanar_higher = torch.abs(rendered_depth - z_buffer) <= depth_tolerance

            update_pixels = valid_pixel & (is_closer | is_coplanar_higher)

            # Zバッファの更新
            z_buffer[update_pixels] = rendered_depth[update_pixels]

            # マスク描画対象として白(True)に設定
            mask_buffer[update_pixels] = True

    mask_np = (mask_buffer.cpu().numpy() * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_np, mode="L")
    return mask_buffer, mask_image


def generate_multiview_masks(
    render_layers_by_idx: Dict[int, List[Tuple[VanillaGaussianModel, bool]]],
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
    non_empty_count = 0
    for idx, cam in enumerate(cameras):
        mask_tensor, mask_image = render_mesh_silhouette_mask_with_zbuffer(
            render_layers_by_idx=render_layers_by_idx,
            cam=cam,
            pipe=pipe,
            depth_tolerance=depth_tolerance,
        )
        mask_filename = os.path.join(output_dir, f"mask_cam_{idx:04d}.png")
        mask_image.save(mask_filename)
        saved_count += 1
        if mask_tensor.sum() > 0:
            non_empty_count += 1
    print(f"  -> 全 {saved_count} カメラ視点の2Dマスクを保存しました (内、メッシュ検出: {non_empty_count} 視点): {output_dir}")


# ==============================================================================
# 4. 既存シーンの重複・内包ガウシアンの自動検出＆くり抜き (Auto-Prune)
# ==============================================================================
def prune_overlapping_gaussians(
    base_gaussian: GaussianModel,
    render_layers_by_idx: Dict[int, List[Tuple[VanillaGaussianModel, bool]]],
    cameras: List,
    pipe: PipelineParams,
    mesh_bounds: Tuple[np.ndarray, np.ndarray],
    prune_margin: float = 0.02,
    depth_tolerance: float = 0.005,
    max_cams_to_check: int = 10,
) -> int:
    """
    配置されたメッシュのバウンディングボックスおよびColmapカメラからのZバッファ・マスク判定に基づき、
    メッシュの内包または真後ろに隠れる既存の不要ガウシアン点群を自動検出して削除する。
    """
    print(f"\n[Step 4: Auto-Prune] メッシュと重複・遮蔽される既存ガウシアンの自動検出を開始...")
    xyz = base_gaussian.get_xyz  # [N, 3]
    N = xyz.shape[0]

    min_b, max_b = mesh_bounds
    # 1. 3Dバウンディングボックスによる粗選抜 (GPUテンソルで高速判定)
    margin_vec = torch.tensor(prune_margin, device=xyz.device)
    min_b_t = torch.from_numpy(min_b).float().to(xyz.device) - margin_vec
    max_b_t = torch.from_numpy(max_b).float().to(xyz.device) + margin_vec

    in_bbox = (
        (xyz[:, 0] >= min_b_t[0]) & (xyz[:, 0] <= max_b_t[0]) &
        (xyz[:, 1] >= min_b_t[1]) & (xyz[:, 1] <= max_b_t[1]) &
        (xyz[:, 2] >= min_b_t[2]) & (xyz[:, 2] <= max_b_t[2])
    )

    candidate_indices = torch.where(in_bbox)[0]
    cand_count = candidate_indices.shape[0]
    print(f"  -> メッシュAABB近傍の候補点数: {cand_count:,} / {N:,} 点")
    if cand_count == 0:
        return 0

    cand_xyz = xyz[candidate_indices]  # [M, 3]
    prune_mask_cand = torch.zeros(cand_count, dtype=torch.bool, device=xyz.device)

    # 2. 代表カメラ視点から見て、メッシュのテクスチャ領域で遮蔽されているかを判定
    step = max(1, len(cameras) // max_cams_to_check)
    cams_to_check = cameras[::step][:max_cams_to_check]

    with torch.no_grad():
        for cam in cams_to_check:
            mesh_mask_buf, _ = render_mesh_silhouette_mask_with_zbuffer(
                render_layers_by_idx=render_layers_by_idx,
                cam=cam,
                pipe=pipe,
                depth_tolerance=depth_tolerance,
            )
            if not torch.any(mesh_mask_buf):
                continue

            R = torch.from_numpy(cam.R).float().to(xyz.device)
            T = torch.from_numpy(cam.T).float().to(xyz.device)
            cand_cam = torch.matmul(cand_xyz, R) + T
            z_cam = cand_cam[:, 2]

            valid_z = z_cam > 0.05
            fx = cam.image_width / (2.0 * np.tan(cam.FoVx / 2.0))
            fy = cam.image_height / (2.0 * np.tan(cam.FoVy / 2.0))
            cx = cam.image_width / 2.0
            cy = cam.image_height / 2.0

            u = (cand_cam[:, 0] * fx / z_cam + cx).long()
            v = (cand_cam[:, 1] * fy / z_cam + cy).long()

            in_screen = valid_z & (u >= 0) & (u < cam.image_width) & (v >= 0) & (v < cam.image_height)
            u_valid = torch.clamp(u, 0, cam.image_width - 1)
            v_valid = torch.clamp(v, 0, cam.image_height - 1)
            hit_mesh_mask = mesh_mask_buf[v_valid, u_valid]

            # メッシュマスク領域に隠れる既存点をマーク
            prune_mask_cand |= (in_screen & hit_mesh_mask)

    pruned_indices = candidate_indices[prune_mask_cand]
    num_pruned = pruned_indices.shape[0]

    if num_pruned > 0:
        full_prune_mask = torch.zeros(N, dtype=torch.bool, device=xyz.device)
        full_prune_mask[pruned_indices] = True
        keep_mask = ~full_prune_mask

        with torch.no_grad():
            base_gaussian._xyz = torch.nn.Parameter(base_gaussian._xyz[keep_mask].requires_grad_(True))
            base_gaussian._features_dc = torch.nn.Parameter(base_gaussian._features_dc[keep_mask].requires_grad_(True))
            base_gaussian._features_rest = torch.nn.Parameter(base_gaussian._features_rest[keep_mask].requires_grad_(True))
            base_gaussian._opacity = torch.nn.Parameter(base_gaussian._opacity[keep_mask].requires_grad_(True))
            base_gaussian._scaling = torch.nn.Parameter(base_gaussian._scaling[keep_mask].requires_grad_(True))
            base_gaussian._rotation = torch.nn.Parameter(base_gaussian._rotation[keep_mask].requires_grad_(True))
            if hasattr(base_gaussian, "max_radii2D") and base_gaussian.max_radii2D is not None:
                if len(base_gaussian.max_radii2D) == N:
                    base_gaussian.max_radii2D = base_gaussian.max_radii2D[keep_mask]
                else:
                    base_gaussian.max_radii2D = torch.zeros(base_gaussian.get_xyz.shape[0], device=xyz.device)
            if hasattr(base_gaussian, "_generation") and base_gaussian._generation is not None:
                if len(base_gaussian._generation) == N:
                    base_gaussian._generation = base_gaussian._generation[keep_mask]
                else:
                    base_gaussian._generation = torch.zeros(base_gaussian.get_xyz.shape[0], dtype=torch.int64, device=xyz.device)

        print(f"  -> 遮蔽・重複していた既存ガウシアン {num_pruned:,} 点を自動削除しました！ (残存: {base_gaussian.get_xyz.shape[0]:,} 点)")
    else:
        print(f"  -> 削除対象となる重複ガウシアンはありませんでした。")

    return num_pruned


# ==============================================================================
# 5. 拡散モデル（InstructPix2Pix等）によるテクスチャ＆境界ブレンディング (Refine)
# ==============================================================================
def refine_with_diffusion(
    gaussian: GaussianModel,
    scene: Scene,
    pipe: PipelineParams,
    prompt: str,
    refine_steps: int = 200,
    per_editing_step: int = 10,
    lr: float = 0.005,
):
    """
    GaussianEditor の InstructPix2Pix / EditGuidance を直接呼び出し、
    結合後のシーンの境界部分を拡散モデルで最適化・ブレンディングする。
    """
    print(f"\n[Step 5: Refine] 拡散モデルによるテクスチャ＆境界ブレンディングを開始...")
    print(f"  -> プロンプト: '{prompt}', 最適化ステップ数: {refine_steps}")

    import random
    from tqdm import trange
    from omegaconf import OmegaConf
    from threestudio.utils.perceptual import PerceptualLoss
    from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
    from threestudio.models.guidance.instructpix2pix_guidance import InstructPix2PixGuidance
    from gaussiansplatting.arguments import OptimizationParams

    # 1. オプティマイザの初期化 (色のファインチューニング)
    opt = OptimizationParams(argparse.ArgumentParser(), max_steps=refine_steps)
    opt.position_lr_init = 0.0
    opt.position_lr_final = 0.0
    opt.scaling_lr = 0.0
    opt.rotation_lr = 0.0
    opt.opacity_lr = 0.0
    opt.feature_lr = lr

    gaussian.training_setup(opt)

    # 2. 拡散モデルとプロンプトプロセッサの初期化
    print("  -> InstructPix2Pix ガイダンスモデルを読み込み中...")
    prompt_processor = StableDiffusionPromptProcessor(
        {
            "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
            "prompt": prompt,
        }
    )
    prompt_utils = prompt_processor()

    ip2p = InstructPix2PixGuidance(
        OmegaConf.create({
            "min_step_percent": 0.02,
            "max_step_percent": 0.98,
        })
    )

    perceptual_loss = PerceptualLoss().eval().to("cuda")
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    train_cams = scene.getTrainCameras()
    H, W = train_cams[0].image_height, train_cams[0].image_width

    # 3. 各カメラ視点の基準画像を事前キャッシュ
    print("  -> 各カメラ視点の初期レンダリング画像を事前取得中...")
    origin_frames = {}
    with torch.no_grad():
        for i, cam in enumerate(train_cams):
            r_pkg = render(cam, gaussian, pipe, bg_color)
            origin_frames[i] = r_pkg["render"].detach().clone()

    edit_frames = {}

    # 4. 最適化ループ
    print(f"  -> 最適化ループを実行中 (0 ~ {refine_steps} steps)...")
    pbar = trange(refine_steps, desc="Refining Gaussian Splatting")
    for step in pbar:
        gaussian.update_learning_rate(step)
        cam_idx = random.randint(0, len(train_cams) - 1)
        cam = train_cams[cam_idx]

        render_pkg = render(cam, gaussian, pipe, bg_color)
        image = render_pkg["render"]

        # InstructPix2Pix で目標画像を生成
        if cam_idx not in edit_frames or (step % per_editing_step == 0):
            with torch.no_grad():
                result = ip2p(
                    image[None].permute(0, 2, 3, 1),
                    origin_frames[cam_idx][None].permute(0, 2, 3, 1),
                    prompt_utils,
                )
                edited_image = torch.nn.functional.interpolate(
                    result["edit_images"].permute(0, 3, 1, 2),
                    (H, W),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                edit_frames[cam_idx] = edited_image.detach()

        gt_image = edit_frames[cam_idx]

        # L1 + Perceptual Loss
        loss_l1 = torch.nn.functional.l1_loss(image, gt_image)
        loss_p = perceptual_loss(image[None].contiguous(), gt_image[None].contiguous()).sum()
        loss = 10.0 * loss_l1 + 10.0 * loss_p

        # アンカーロス（幾何形状・スケールの安定化）
        if hasattr(gaussian, "anchor_loss"):
            anchor_out = gaussian.anchor_loss()
            loss += 5.0 * anchor_out.get('loss_anchor_geo', 0.0) + 5.0 * anchor_out.get('loss_anchor_scale', 0.0)

        loss.backward()
        gaussian.optimizer.step()
        gaussian.optimizer.zero_grad(set_to_none=True)

        if step % 10 == 0:
            pbar.set_postfix({"L1": f"{loss_l1.item():.4f}", "Loss": f"{loss.item():.4f}"})

    print(f"  -> 拡散モデルによる境界ブレンディングが完了しました！")


# ==============================================================================
# 6. シーンの結合と保存
# ==============================================================================
def merge_and_save_scene(
    base_gaussian: GaussianModel,
    obj_gaussian: VanillaGaussianModel,
    output_merged_ply: str,
):
    print(f"\n[Step 6: Merge] シーンの結合と保存処理...")
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
# ==============================================================================
# メイン実行エントリポイント
# ==============================================================================
def find_mesh_file_in_dir(dir_path: str) -> Optional[str]:
    """ディレクトリ内からGLB/GLTF/OBJファイルを自動検索する"""
    if not os.path.exists(dir_path):
        return None
    for root, _, files in os.walk(dir_path):
        for f in sorted(files):
            if f.lower().endswith((".glb", ".gltf", ".obj")):
                return os.path.join(root, f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="【Mesh2Gaussian】Mesh2Gaussian_dataフォルダを介して最小限の引数で実行できる3DGS統合＆最適化プログラム"
    )
    # ベースデータディレクトリ
    parser.add_argument("--data_dir", type=str, default="Mesh2Gaussian_data", help="入出力の基準となるフォルダ (デフォルト: Mesh2Gaussian_data)")

    # 入力パス (未指定時は data_dir 配下から自動検出)
    parser.add_argument("--mesh", "-m", type=str, default=None, help="配置済み3Dメッシュファイルパス (.glb推奨)。未指定時は data_dir/input/mesh/ から自動検出")
    parser.add_argument("--source_path", "-s", type=str, default=None, help="Colmapデータセットのパス。未指定時は data_dir/input/colmap/ を使用")
    parser.add_argument("--model_path", type=str, default=None, help="学習済み3DGSモデルのパス。未指定時は data_dir/input/model/ または source_path")
    parser.add_argument("--iteration", type=int, default=-1, help="読み込む学習イテレーション番号 (デフォルト: -1 最適/最新)")

    # 出力パス (未指定時は data_dir/output/ 配下に自動設定)
    parser.add_argument("--output_ply", "-o", type=str, default=None, help="出力先PLYファイル名 (未指定時: data_dir/output/merged_refined.ply)")
    parser.add_argument("--save_mask", type=str, default=None, help="基準カメラの2Dマスク保存先 (未指定時: data_dir/output/masks/mask_cam0.png)")
    parser.add_argument("--multiview_mask_dir", type=str, default=None, help="全カメラ2Dマスク保存先フォルダ (未指定時: data_dir/output/masks/multiview/)")
    parser.add_argument("--no_multiview", action="store_true", default=False, help="全カメラ視点2Dマスクの一括生成を無効化する")

    # メッシュ変換＆レンダリング設定
    parser.add_argument("--num_points", "-n", type=int, default=200000, help="サンプリングするガウシアン点数 (デフォルト: 200,000)")
    parser.add_argument("--mesh_log_path", type=str, default=None, help="メッシュ解析ログの保存先パス (未指定時: output/logs/mesh_log.txt)")
    parser.add_argument("--cam_id", type=int, default=0, help="2Dマスクをレンダリング保存する基準カメラのID (デフォルト: 0)")
    parser.add_argument("--depth_tolerance", type=float, default=0.005, help="同一平面判定トレランス (Z-fighting解消用, デフォルト: 0.005)")

    # 座標系補正 (必要に応じて回転角度を指定。デフォルトは回転補正なし: 0.0)
    parser.add_argument("--rotate_x", "--rotate-x", type=float, default=0.0, help="メッシュに適用するX軸回転角度(度)。デフォルト: 0.0")
    parser.add_argument("--rotate_y", "--rotate-y", type=float, default=0.0, help="メッシュに適用するY軸回転角度(度)。デフォルト: 0.0")
    parser.add_argument("--rotate_z", "--rotate-z", type=float, default=0.0, help="メッシュに適用するZ軸回転角度(度)。デフォルト: 0.0")

    # 自動処理フラグ (デフォルトで True。スキップしたい場合のみ --no_* を指定)
    parser.add_argument("--no_prune", "--no-prune", action="store_true", default=False, help="不要な重複点群の自動削除を無効化する")
    parser.add_argument("--prune_margin", "--prune-margin", type=float, default=0.02, help="重複点判定時のバウンディングボックス／深度マージン (デフォルト: 0.02)")

    parser.add_argument("--no_refine", "--no-refine", action="store_true", default=False, help="拡散モデルによる境界ブレンディングを無効化する")
    parser.add_argument("--prompt", type=str, default="a photorealistic seamless integration, extremely detailed", help="境界ブレンディング用プロンプト")
    parser.add_argument("--refine_steps", "--refine-steps", type=int, default=200, help="境界ブレンディングの最適化ステップ数 (デフォルト: 200)")
    parser.add_argument("--per_editing_step", "--per-editing-step", type=int, default=10, help="拡散モデルによる目標画像更新インターバル (デフォルト: 10)")
    parser.add_argument("--lr", type=float, default=0.005, help="ファインチューニングの学習率 (デフォルト: 0.005)")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[*] 注意: 未知の引数は無視されました: {unknown}")

    # --------------------------------------------------------------------------
    # 0. Mesh2Gaussian_data フォルダ構造の自動準備 ＆ パス自動解決
    # --------------------------------------------------------------------------
    base_dir = args.data_dir
    input_mesh_dir = os.path.join(base_dir, "input", "mesh")
    input_colmap_dir = os.path.join(base_dir, "input", "colmap")
    input_model_dir = os.path.join(base_dir, "input", "model")
    output_dir = os.path.join(base_dir, "output")
    output_masks_dir = os.path.join(output_dir, "masks")

    os.makedirs(input_mesh_dir, exist_ok=True)
    os.makedirs(input_colmap_dir, exist_ok=True)
    os.makedirs(input_model_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    # 1) メッシュファイルのパス解決
    mesh_path = args.mesh
    if mesh_path is None or not os.path.exists(mesh_path):
        detected = find_mesh_file_in_dir(input_mesh_dir) or find_mesh_file_in_dir(os.path.join(base_dir, "input"))
        if detected:
            mesh_path = detected
            print(f"[*] メッシュファイルを自動検出しました: {mesh_path}")
        else:
            raise FileNotFoundError(
                f"\n[Error] 3Dメッシュファイルが見つかりません。\n"
                f"コマンドラインで `--mesh <パス>` を指定するか、\n"
                f"`{input_mesh_dir}` フォルダ内に .glb ファイルを配置してください。"
            )

    # 2) Colmapデータセットのパス解決
    source_path = args.source_path
    if source_path is None or not os.path.exists(source_path):
        if os.path.exists(os.path.join(input_colmap_dir, "sparse")):
            source_path = input_colmap_dir
        else:
            raise FileNotFoundError(
                f"\n[Error] Colmapデータセットが見つかりません。\n"
                f"コマンドラインで `--source_path <パス>` を指定するか、\n"
                f"`{input_colmap_dir}` フォルダ内に sparse/ や images/ を配置してください。"
            )

    # 3) モデルパスの解決
    model_path = args.model_path
    if model_path is None or not os.path.exists(model_path):
        resolved = resolve_model_path(input_model_dir)
        if resolved:
            model_path = resolved
            print(f"[*] 学習済み3DGSモデルを自動検出しました: {model_path}")
        elif os.path.exists(os.path.join(source_path, "point_cloud")):
            model_path = source_path
        else:
            model_path = input_model_dir
            print(f"[*] 学習済みモデルパスを設定: {model_path}")

    # 4) 出力パスの解決 (すべて Mesh2Gaussian_data/output/ 配下に統一)
    auto_prune_enabled = not args.no_prune
    refine_enabled = not args.no_refine

    default_ply_name = "merged_refined.ply" if refine_enabled else "merged.ply"
    output_ply = args.output_ply if args.output_ply else os.path.join(output_dir, default_ply_name)
    save_mask = args.save_mask if args.save_mask else os.path.join(output_masks_dir, "mask_cam0.png")
    multiview_enabled = not args.no_multiview
    multiview_mask_dir = args.multiview_mask_dir if args.multiview_mask_dir else os.path.join(output_masks_dir, "multiview")

    mesh_log_path = args.mesh_log_path if args.mesh_log_path else os.path.join(output_dir, "logs", "mesh_log.txt")

    # 回転補正角度 (デフォルト: 回転なし 0.0)
    rot_x = args.rotate_x
    rot_y = args.rotate_y
    rot_z = args.rotate_z

    print("=" * 70)
    print("【Mesh2Gaussian 実行構成】")
    print(f"  ・データ基底フォルダ : {os.path.abspath(base_dir)}")
    print(f"  ・3Dメッシュ入力     : {mesh_path}")
    print(f"  ・Colmapデータセット : {source_path}")
    print(f"  ・3DGSモデル入力     : {model_path}")
    print(f"  ・座標系回転補正     : X={rot_x}°, Y={rot_y}°, Z={rot_z}°")
    print(f"  ・出力PLYファイル    : {output_ply}")
    print(f"  ・単一2Dマスク保存先 : {save_mask}")
    print(f"  ・全カメラマスク保存 : {'有効 (フォルダ: ' + multiview_mask_dir + ')' if multiview_enabled else '無効'}")
    print(f"  ・メッシュ解析ログ   : {mesh_log_path}")
    print(f"  ・除外対象メッシュ   : 一番最後のメッシュ (末尾)")
    print(f"  ・不要点群自動削除   : {'有効' if auto_prune_enabled else '無効'}")
    print(f"  ・拡散モデルRefine   : {'有効 (Steps: ' + str(args.refine_steps) + ')' if refine_enabled else '無効'}")
    print("=" * 70)


    pipe = PipelineParams(argparse.ArgumentParser())

    # --------------------------------------------------------------------------
    # 1. GLBメッシュのガウシアン化 & レイヤー分類 & バウンディングボックス取得
    # --------------------------------------------------------------------------
    temp_gs_path = os.path.join(output_dir, "temp_mesh_gs.ply")
    full_gaussian, render_layers_by_idx, mesh_bounds = convert_mesh_to_gaussian(
        mesh_path=mesh_path,
        output_gs_path=temp_gs_path,
        num_points=args.num_points,
        rotate_x=rot_x,
        rotate_y=rot_y,
        rotate_z=rot_z,
        mesh_log_path=mesh_log_path,
    )

    # --------------------------------------------------------------------------
    # 2. 既存のベースシーンおよびカメラの読み込み
    # --------------------------------------------------------------------------
    base_gaussian, scene = load_base_scene(
        source_path=source_path,
        model_path=model_path,
        iteration=args.iteration,
    )

    train_cams = scene.getTrainCameras()
    cam = train_cams[min(args.cam_id, len(train_cams) - 1)]

    # --------------------------------------------------------------------------
    # 3. インテリジェント遮蔽 ＆ 内部Zバッファ合成による2Dマスク生成 (全視点デフォルト有効)
    # --------------------------------------------------------------------------
    if save_mask:
        _, mask_img = render_mesh_silhouette_mask_with_zbuffer(
            render_layers_by_idx=render_layers_by_idx,
            cam=cam,
            pipe=pipe,
            depth_tolerance=args.depth_tolerance,
        )
        os.makedirs(os.path.dirname(os.path.abspath(save_mask)), exist_ok=True)
        mask_img.save(save_mask)
        print(f"\n[Mask] 基準カメラ {args.cam_id} の2Dマスクを保存しました: {save_mask}")

    if multiview_enabled and multiview_mask_dir:
        generate_multiview_masks(
            render_layers_by_idx=render_layers_by_idx,
            cameras=train_cams,
            output_dir=multiview_mask_dir,
            pipe=pipe,
            depth_tolerance=args.depth_tolerance,
        )

    # --------------------------------------------------------------------------
    # 4. 既存の被る不要点群の自動くり抜き (Auto-Prune: デフォルト有効)
    # --------------------------------------------------------------------------
    if auto_prune_enabled:
        prune_overlapping_gaussians(
            base_gaussian=base_gaussian,
            render_layers_by_idx=render_layers_by_idx,
            cameras=train_cams,
            pipe=pipe,
            mesh_bounds=mesh_bounds,
            prune_margin=args.prune_margin,
            depth_tolerance=args.depth_tolerance,
        )

    # --------------------------------------------------------------------------
    # 5. シーンの結合
    # --------------------------------------------------------------------------
    print(f"\n[Step 5] メッシュガウシアンをベースシーンに結合します...")
    if base_gaussian.optimizer is None:
        from gaussiansplatting.arguments import OptimizationParams
        opt = OptimizationParams(argparse.ArgumentParser(), max_steps=args.refine_steps if refine_enabled else 1000)
        if hasattr(scene, "cameras_extent"):
            base_gaussian.spatial_lr_scale = scene.cameras_extent
        base_gaussian.training_setup(opt)

    base_gaussian.concat_gaussians(full_gaussian)

    # --------------------------------------------------------------------------
    # 6. 拡散モデルによる境界ブレンディング・最適化 (Refine: デフォルト有効)
    # --------------------------------------------------------------------------
    if refine_enabled:
        refine_with_diffusion(
            gaussian=base_gaussian,
            scene=scene,
            pipe=pipe,
            prompt=args.prompt,
            refine_steps=args.refine_steps,
            per_editing_step=args.per_editing_step,
            lr=args.lr,
        )

    # --------------------------------------------------------------------------
    # 7. 最終シーンの保存
    # --------------------------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(output_ply)), exist_ok=True)
    base_gaussian.save_ply(output_ply)
    print(f"\n[Success] すべての処理が完了しました！")
    print(f"  -> 最終統合PLYファイル: {os.path.abspath(output_ply)}")
    print(f"  -> 出力マスクフォルダ: {os.path.abspath(output_masks_dir)}\n")

    # 一時ファイルの削除
    if os.path.exists(temp_gs_path):
        os.remove(temp_gs_path)


if __name__ == "__main__":
    main()


