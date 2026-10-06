"""
mesh_to_gaussian.py
メッシュモデル（OBJ, PLY, STL, GLB, GLTF等）を入力し、
3D Gaussian Splatting (3DGS) の点群に変換してPLYファイルとして出力するプログラム。

GaussianEditor の train_from_mesh.py, gaussian_model.py, threestudio/utils/mesh.py
からメッシュのガウシアン化およびPLY保存のコアロジックを抽出・集約しています。
"""

import os
import sys
import argparse
import math
from typing import Optional, Tuple

import numpy as np
import torch
from plyfile import PlyData, PlyElement

try:
    import trimesh
    from PIL import Image
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

# CUDA版の近傍探索（利用可能な場合）
try:
    from simple_knn._C import distCUDA2
    HAS_SIMPLE_KNN = True
except ImportError:
    HAS_SIMPLE_KNN = False

# CPU版の近傍探索フォールバック
try:
    from scipy.spatial import cKDTree
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ==============================================================================
# 0. 数学・3DGS変換ユーティリティ
# ==============================================================================

# 球面調和関数 (SH) 0次基底の係数: Y_0^0 = 1 / (2 * sqrt(pi))
SH_C0 = 0.28209479177387814

def rgb_to_sh0(rgb: np.ndarray) -> np.ndarray:
    """RGBカラー [0, 1] を 3DGS の 0次SH係数 (f_dc) に変換"""
    return (rgb - 0.5) / SH_C0

def sh0_to_rgb(sh0: np.ndarray) -> np.ndarray:
    """3DGS の 0次SH係数 (f_dc) を RGBカラー [0, 1] に変換"""
    return np.clip(sh0 * SH_C0 + 0.5, 0.0, 1.0)

def inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    """シグモイド関数の逆関数 (logit)"""
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


# ==============================================================================
# 1. メッシュ読み込みおよび表面点群サンプリング (OBJ, PLY, GLTF/GLB, STL対応)
# ==============================================================================

def sample_colors_from_submesh(
    mesh: trimesh.Trimesh,
    points: np.ndarray,
    face_indices: np.ndarray,
    external_texture_img: Optional[Image.Image] = None,
    default_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
) -> np.ndarray:
    """
    サブメッシュからUV・テクスチャ・頂点カラー・マテリアル色を考慮して色をサンプリング。
    GLTFのPBRマテリアル (baseColorTexture, baseColorFactor) にも対応。
    """
    n_pts = points.shape[0]
    if n_pts == 0:
        return np.empty((0, 3), dtype=np.float32)

    # 1. テクスチャ画像の探索
    tex_img = external_texture_img
    mat = getattr(mesh.visual, "material", None)
    base_factor = None

    if tex_img is None and mat is not None:
        # 一般的なテクスチャ
        if hasattr(mat, "image") and mat.image is not None:
            tex_img = mat.image.convert("RGB")
        # GLTF PBR Material の baseColorTexture
        elif hasattr(mat, "baseColorTexture") and mat.baseColorTexture is not None:
            tex_img = mat.baseColorTexture.convert("RGB")

        # baseColorFactor の取得
        if hasattr(mat, "baseColorFactor") and mat.baseColorFactor is not None:
            bf = np.asarray(mat.baseColorFactor, dtype=np.float32)[:3]
            base_factor = bf
        elif hasattr(mat, "diffuse") and mat.diffuse is not None:
            bf = np.asarray(mat.diffuse, dtype=np.float32)[:3]
            if np.max(bf) > 1.0:
                bf = bf / 255.0
            base_factor = bf

    # 2. テクスチャ画像 + UV座標からのサンプリング
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

            # テクスチャサンプリング (反転・リピート考慮)
            u = np.clip(sampled_uvs[:, 0] % 1.0, 0.0, 1.0) * (tex_w - 1)
            v = np.clip((1.0 - (sampled_uvs[:, 1] % 1.0)), 0.0, 1.0) * (tex_h - 1)
            u_idx = np.clip(np.round(u).astype(int), 0, tex_w - 1)
            v_idx = np.clip(np.round(v).astype(int), 0, tex_h - 1)
            colors = tex_np[v_idx, u_idx, :3]

            if base_factor is not None:
                colors = colors * base_factor

            return colors.astype(np.float32)
        except Exception as e:
            print(f"[Warning] テクスチャサンプリング失敗: {e}. 代替カラーを試行します。")

    # 3. 頂点カラーからのサンプリング
    if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None and len(mesh.visual.vertex_colors) > 0:
        try:
            v_colors = mesh.visual.vertex_colors[:, :3].astype(np.float32)
            if np.max(v_colors) > 1.0:
                v_colors = v_colors / 255.0
            faces = mesh.faces[face_indices]
            tri_verts = mesh.vertices[faces]
            barycentric = trimesh.triangles.points_to_barycentric(tri_verts, points)
            colors = np.sum(v_colors[faces] * barycentric[:, :, None], axis=1)
            return colors.astype(np.float32)
        except Exception as e:
            print(f"[Warning] 頂点カラー取得失敗: {e}")

    # 4. マテリアルの単色（baseColorFactor または diffuse）
    if base_factor is not None:
        return np.full((n_pts, 3), base_factor, dtype=np.float32)

    # 5. デフォルト色
    return np.full((n_pts, 3), default_color, dtype=np.float32)


def sample_mesh_surface(
    mesh_path: str,
    num_points: int = 200000,
    external_texture_path: Optional[str] = None,
    color_mode: str = "auto",
    default_color: Tuple[float, float, float] = (0.8, 0.8, 0.8)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    メッシュ (OBJ, PLY, STL, GLTF/GLB 等) から表面点群・法線・色を一様サンプリングする。
    GLTFなどの階層ノードやトランスフォーム、複数マテリアルに対応。

    Returns:
        points: (N, 3) 頂点座標
        normals: (N, 3) 頂点法線
        colors: (N, 3) RGBカラー [0, 1]
    """
    if not HAS_TRIMESH:
        raise ImportError("メッシュ読み込みには trimesh が必要です。'pip install trimesh' を実行してください。")

    # 外部テクスチャが指定されている場合はロード
    ext_tex_img = None
    if external_texture_path and os.path.exists(external_texture_path):
        ext_tex_img = Image.open(external_texture_path).convert("RGB")

    # メッシュ読み込み (process=False でテクスチャやUVの自動削除を防止)
    loaded = trimesh.load(mesh_path, process=False)

    # GLTF / GLB などの Scene オブジェクトの場合、各ノードのトランスフォームを適用して個別展開
    submeshes = []
    if isinstance(loaded, trimesh.Scene):
        # dump() で各ジオメトリインスタンスのワールド変換行列を適用した Trimesh リストを取得
        dumped = loaded.dump()
        for m in dumped:
            if isinstance(m, trimesh.Trimesh) and len(m.vertices) > 0 and len(m.faces) > 0:
                submeshes.append(m)
    elif isinstance(loaded, trimesh.Trimesh):
        submeshes.append(loaded)
    else:
        raise ValueError(f"サポートされていない形式または空のデータです: {type(loaded)}")

    if len(submeshes) == 0:
        raise ValueError("有効なメッシュジオメトリが見つかりませんでした。")

    # 各サブメッシュの表面積を計算してサンプリング点数を配分
    areas = np.array([max(m.area, 1e-9) for m in submeshes], dtype=np.float64)
    total_area = np.sum(areas)
    if total_area <= 0:
        weights = np.ones(len(submeshes)) / len(submeshes)
    else:
        weights = areas / total_area

    # 各サブメッシュのサンプリング点数（合計が num_points になるよう按分）
    sub_counts = np.round(weights * num_points).astype(int)
    # 最低1点または合計点数の調整
    diff = num_points - np.sum(sub_counts)
    if diff != 0:
        max_idx = np.argmax(areas)
        sub_counts[max_idx] += diff

    all_points = []
    all_normals = []
    all_colors = []

    for mesh, count in zip(submeshes, sub_counts):
        if count <= 0:
            continue

        # 表面サンプリング
        pts, face_indices = trimesh.sample.sample_surface(mesh, count)

        # 法線ベクトルの取得
        if len(mesh.face_normals) > 0 and len(face_indices) > 0:
            norms = mesh.face_normals[face_indices]
        else:
            norms = np.zeros_like(pts)
            norms[:, 2] = 1.0

        # 色の取得
        if color_mode == "random":
            cols = np.random.uniform(0.0, 1.0, size=(count, 3)).astype(np.float32)
        elif color_mode == "white":
            cols = np.ones((count, 3), dtype=np.float32)
        else:
            cols = sample_colors_from_submesh(
                mesh=mesh,
                points=pts,
                face_indices=face_indices,
                external_texture_img=ext_tex_img,
                default_color=default_color
            )

        all_points.append(pts)
        all_normals.append(norms)
        all_colors.append(cols)

    points = np.concatenate(all_points, axis=0).astype(np.float32)
    normals = np.concatenate(all_normals, axis=0).astype(np.float32)
    colors = np.concatenate(all_colors, axis=0).astype(np.float32)

    # 点数が微小にずれた場合のカット / パディング
    if points.shape[0] > num_points:
        points = points[:num_points]
        normals = normals[:num_points]
        colors = colors[:num_points]

    return points, normals, colors


# ==============================================================================
# 2. ガウシアンスケールの計算 (k-NN距離)
# ==============================================================================

def compute_gaussian_scales(
    points: np.ndarray,
    normals: Optional[np.ndarray] = None,
    flat_disks: bool = False,
    thickness_ratio: float = 0.15
) -> np.ndarray:
    """
    点群の各点について、近傍点との平均距離からスケール(log scale)を計算する。
    GaussianEditor / 3DGS の create_from_pcd と同等のロジック。
    """
    num_points = points.shape[0]

    # 1. CUDA 版 distCUDA2 が利用可能な場合（超高速）
    if HAS_SIMPLE_KNN and torch.cuda.is_available():
        pts_tensor = torch.from_numpy(points).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(pts_tensor), 0.0000001)
        mean_dist = torch.sqrt(dist2).cpu().numpy()
    # 2. SciPy cKDTree によるフォールバック
    elif HAS_SCIPY:
        tree = cKDTree(points)
        # 自身を含む近傍4点を取得し、自身を除く3点の平均距離を算出
        dists, _ = tree.query(points, k=4, workers=-1)
        mean_dist = np.mean(dists[:, 1:], axis=1)
        mean_dist = np.maximum(mean_dist, 1e-7)
    # 3. PyTorch (CPU/GPU) による簡易フォールバック（点数が多い場合はサブサンプリング）
    else:
        pts_tensor = torch.from_numpy(points).float()
        # メモリ節約のためチャンク分割
        chunk_size = 2048
        mean_dists = []
        for i in range(0, num_points, chunk_size):
            chunk = pts_tensor[i:i+chunk_size]
            dist_mat = torch.cdist(chunk, pts_tensor)
            topk_dists, _ = torch.topk(dist_mat, k=4, largest=False)
            chunk_mean = torch.mean(topk_dists[:, 1:], dim=1)
            mean_dists.append(chunk_mean.numpy())
        mean_dist = np.concatenate(mean_dists, axis=0)
        mean_dist = np.maximum(mean_dist, 1e-7)

    if flat_disks and normals is not None:
        # メッシュ表面に沿った薄型楕円体（ディスク状）
        # 接平面方向は接するサイズ、法線方向は薄く設定
        s_tangent = np.log(mean_dist)[:, None]
        s_normal = np.log(mean_dist * thickness_ratio)[:, None]
        scales = np.concatenate([s_tangent, s_tangent, s_normal], axis=1)
    else:
        # 等方性スケール (3DGS標準の初期化)
        scales = np.log(mean_dist)[:, None].repeat(3, axis=1)

    return scales.astype(np.float32)


# ==============================================================================
# 3. 回転クォータニオンの計算
# ==============================================================================

def compute_gaussian_rotations(
    normals: Optional[np.ndarray] = None,
    align_to_normals: bool = False
) -> np.ndarray:
    """
    ガウシアンの回転クォータニオン [w, x, y, z] を生成。
    """
    num_points = normals.shape[0] if normals is not None else 0

    if not align_to_normals or normals is None:
        # 3DGS標準: 単位クォータニオン [1, 0, 0, 0]
        rots = np.zeros((num_points, 4), dtype=np.float32)
        rots[:, 0] = 1.0
        return rots

    # 法線ベクトルにZ軸を揃える回転クォータニオンを算出
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    n = normals / norms

    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    dot = np.sum(n * z_axis, axis=1)
    cross = np.cross(z_axis, n)

    # クォータニオン q = [w, x, y, z]
    w = 1.0 + dot
    rots = np.concatenate([w[:, None], cross], axis=1)

    # 特異点対応（逆向きベクトル）
    opp_mask = (dot < -0.9999)
    if np.any(opp_mask):
        rots[opp_mask] = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    q_norm = np.linalg.norm(rots, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    rots = rots / q_norm

    return rots.astype(np.float32)


# ==============================================================================
# 4. PLYファイルの構築と保存
# ==============================================================================

def save_gaussian_ply(
    save_path: str,
    xyz: np.ndarray,
    colors: np.ndarray,
    scales: np.ndarray,
    rots: np.ndarray,
    opacities: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    sh_degree: int = 3
):
    """
    ガウシアンパラメータを 3DGS 標準バイナリ PLY ファイルとして保存する。
    GaussianModel.save_ply と完全な互換性を持つフォーマットです。
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    num_points = xyz.shape[0]

    # 1. 法線
    if normals is None:
        normals = np.zeros_like(xyz)

    # 2. 0次SH係数 (f_dc_0, f_dc_1, f_dc_2)
    f_dc = rgb_to_sh0(colors)

    # 3. 高次SH係数 (f_rest_*)
    num_sh_bases = (sh_degree + 1) ** 2
    num_rest_channels = (num_sh_bases - 1) * 3
    if num_rest_channels > 0:
        f_rest = np.zeros((num_points, num_rest_channels), dtype=np.float32)
    else:
        f_rest = np.empty((num_points, 0), dtype=np.float32)

    # 4. 不透明度 (opacity)
    if opacities is None:
        # GaussianEditor の create_from_pcd 同様、1.0（または inverse_sigmoid(0.99)）
        # 3DGSのレンダラでは sigmoid(opacity) が適用されるため、実効値 ~0.99 に相当する値
        opacities = np.full((num_points, 1), 4.595, dtype=np.float32)
    elif opacities.ndim == 1:
        opacities = opacities[:, None]

    # PLY属性リストの構築
    attrs = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attrs.append(f"f_dc_{i}")
    for i in range(num_rest_channels):
        attrs.append(f"f_rest_{i}")
    attrs.append("opacity")
    for i in range(3):
        attrs.append(f"scale_{i}")
    for i in range(4):
        attrs.append(f"rot_{i}")

    # 構造化配列 (structured array) の定義
    dtype_full = [(name, "f4") for name in attrs]
    elements = np.empty(num_points, dtype=dtype_full)

    # 属性データの結合
    combined = np.concatenate(
        [xyz, normals, f_dc, f_rest, opacities, scales, rots],
        axis=1
    )

    elements[:] = list(map(tuple, combined))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(save_path)
    print(f"[Success] 3DGS PLYファイルを保存しました: {save_path} (点数: {num_points:,})")


# ==============================================================================
# 5. メイン処理関数
# ==============================================================================

def mesh_to_gaussian_ply(
    mesh_path: str,
    output_ply_path: str,
    num_points: int = 200000,
    texture_path: Optional[str] = None,
    color_mode: str = "auto",
    default_color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
    flat_disks: bool = False,
    sh_degree: int = 3
):
    """
    メッシュファイルを読み込み、表面サンプリングを行って3DGSのPLYファイルを出力する。

    Args:
        mesh_path: 入力メッシュファイルパス (.obj, .ply, .stl, .glb, .gltf等)
        output_ply_path: 出力先PLYファイルパス (.ply)
        num_points: ガウシアン化するサンプリング点数 (デフォルト: 200,000)
        texture_path: 外部テクスチャ画像のパス (任意)
        color_mode: カラー取得モード ('auto', 'texture', 'random', 'white')
        default_color: 色が見つからない場合のデフォルトRGB (0.0〜1.0)
        flat_disks: Trueの場合、メッシュ表面に沿ったディスク状のガウシアンを生成
        sh_degree: 球面調和関数の次数 (デフォルト: 3)
    """
    print(f"=== メッシュのガウシアン化開始 ===")
    print(f"  入力メッシュ: {mesh_path}")
    print(f"  サンプリング点数: {num_points:,}")
    print(f"  カラーモード: {color_mode}")
    if texture_path:
        print(f"  指定テクスチャ: {texture_path}")

    # 1. メッシュ表面のサンプリング
    points, normals, colors = sample_mesh_surface(
        mesh_path=mesh_path,
        num_points=num_points,
        external_texture_path=texture_path,
        color_mode=color_mode,
        default_color=default_color
    )
    print(f"  サンプリング完了: 点数 = {points.shape[0]:,}")

    # 2. スケールの計算 (k-NN距離)
    scales = compute_gaussian_scales(
        points=points,
        normals=normals,
        flat_disks=flat_disks
    )

    # 3. 回転クォータニオンの計算
    rots = compute_gaussian_rotations(
        normals=normals,
        align_to_normals=flat_disks
    )

    # 4. PLYファイル保存
    save_gaussian_ply(
        save_path=output_ply_path,
        xyz=points,
        colors=colors,
        scales=scales,
        rots=rots,
        normals=normals,
        sh_degree=sh_degree
    )


# ==============================================================================
# 6. コマンドライン引数実行 (CLI)
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="メッシュモデルを入力し、3D Gaussian Splatting (3DGS) のPLYファイルを出力するスクリプト"
    )
    parser.add_argument(
        "--mesh", "-m", type=str, required=True,
        help="入力メッシュモデルのパス (.obj, .ply, .stl, .glb, .gltf など)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="出力先PLYファイルのパス (指定しない場合は <メッシュ名>_gaussian.ply)"
    )
    parser.add_argument(
        "--num_points", "-n", type=int, default=200000,
        help="生成するガウシアンの点数 (デフォルト: 200,000)"
    )
    parser.add_argument(
        "--texture", "-t", type=str, default=None,
        help="テクスチャ画像ファイルのパス (任意)"
    )
    parser.add_argument(
        "--color_mode", type=str, default="auto",
        choices=["auto", "texture", "random", "white"],
        help="色の決定方式: auto (テクスチャ/頂点カラー優先), random (ランダム色), white (白色)"
    )
    parser.add_argument(
        "--flat_disks", action="store_true",
        help="メッシュ表面に沿った薄いディスク型ガウシアンにするフラグ"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3,
        help="球面調和関数の最大次数 (デフォルト: 3)"
    )

    args = parser.parse_args()

    # 出力パスのデフォルト設定
    if args.output is None:
        base, _ = os.path.splitext(args.mesh)
        output_ply = f"{base}_gaussian.ply"
    else:
        output_ply = args.output

    if args.color_mode == "white":
        default_color = (1.0, 1.0, 1.0)
    else:
        default_color = (0.8, 0.8, 0.8)

    mesh_to_gaussian_ply(
        mesh_path=args.mesh,
        output_ply_path=output_ply,
        num_points=args.num_points,
        texture_path=args.texture,
        color_mode=args.color_mode,
        default_color=default_color,
        flat_disks=args.flat_disks,
        sh_degree=args.sh_degree
    )


if __name__ == "__main__":
    main()
