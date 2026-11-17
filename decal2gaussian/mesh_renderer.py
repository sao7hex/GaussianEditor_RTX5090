import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh
from PIL import Image


def colmap_cam_to_opengl_pose(cam) -> np.ndarray:
    """
    3DGS / COLMAP の Camera オブジェクトから OpenGL (pyrender) 用の Camera-to-World 行列 (4x4) を計算する。
    COLMAP座標系 (X-Right, Y-Down, Z-Forward) -> OpenGL座標系 (X-Right, Y-Up, Z-Backward)
    """
    w2c = cam.world_view_transform.detach().cpu().numpy().T
    c2w = np.linalg.inv(w2c)
    opengl_fix = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return (c2w @ opengl_fix).astype(np.float32)


def _has_transparent_texture(geom: trimesh.Trimesh) -> bool:
    """メッシュのテクスチャまたはマテリアルに透明領域が含まれているか判定する"""
    mat = getattr(getattr(geom, "visual", None), "material", None)
    if mat is None:
        return False

    base_factor = getattr(mat, "baseColorFactor", None)
    if base_factor is not None and len(base_factor) >= 4 and float(base_factor[3]) < 1.0:
        return True

    tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if tex is not None:
        arr = np.asarray(tex)
        if arr.ndim == 3 and arr.shape[2] >= 4 and int(arr[:, :, 3].min()) < 255:
            return True
    return False


def _configure_transparency(
    pr_mesh,
    geom: trimesh.Trimesh,
    alpha_mode: str = "MASK",
    alpha_cutoff: float = 0.05,
) -> None:
    """透明領域を透明として出力するため、pyrender マテリアルの alphaMode を設定する"""
    has_alpha = _has_transparent_texture(geom)
    for prim in getattr(pr_mesh, "primitives", []):
        mat = getattr(prim, "material", None)
        if mat is None:
            continue
        current_mode = getattr(mat, "alphaMode", None)
        if has_alpha or current_mode in ("MASK", "BLEND"):
            mat.alphaMode = alpha_mode
            if alpha_mode == "MASK":
                mat.alphaCutoff = float(alpha_cutoff)


def render_glb_from_cameras(
    meshes: List[Tuple[str, trimesh.Trimesh]],
    cameras: List,
    output_dir: str,
    cam_indices: Optional[List[int]] = None,
    flat: bool = True,
    alpha_mode: str = "MASK",
    alpha_cutoff: float = 0.05,
) -> List[Path]:
    """
    読み込んだ GLB メッシュ群を COLMAP カメラ視点からオフスクリーンレンダリングし、RGBA PNG として保存する。
    """
    import pyrender

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pr_scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[1.0, 1.0, 1.0],
    )

    added_count = 0
    for node_name, geom in meshes:
        if len(geom.vertices) == 0 or len(geom.faces) == 0:
            continue
        try:
            pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
            _configure_transparency(
                pr_mesh,
                geom,
                alpha_mode=alpha_mode,
                alpha_cutoff=alpha_cutoff,
            )
            pr_scene.add(pr_mesh, name=node_name)
            added_count += 1
        except Exception as e:
            print(f"  [Warning] メッシュ '{node_name}' の pyrender 変換をスキップしました: {e}")

    if added_count == 0:
        raise RuntimeError("レンダリング可能な有効メッシュがありません。")

    if not flat:
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        pr_scene.add(light, pose=np.eye(4))

    target_indices = cam_indices if cam_indices is not None else list(range(len(cameras)))
    flags = pyrender.RenderFlags.RGBA
    if flat:
        flags |= pyrender.RenderFlags.FLAT

    renderer = None
    current_size = None
    saved_paths: List[Path] = []

    try:
        for idx in target_indices:
            if idx < 0 or idx >= len(cameras):
                print(f"  [Warning] cam_id={idx} は範囲外のためスキップします (0..{len(cameras) - 1})")
                continue

            cam = cameras[idx]
            H, W = int(cam.image_height), int(cam.image_width)
            if current_size != (W, H):
                if renderer is not None:
                    renderer.delete()
                renderer = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
                current_size = (W, H)

            aspect = math.tan(cam.FoVx / 2.0) / math.tan(cam.FoVy / 2.0)
            pr_cam = pyrender.PerspectiveCamera(
                yfov=float(cam.FoVy),
                aspectRatio=float(aspect),
                znear=float(cam.znear),
                zfar=float(cam.zfar),
            )
            cam_pose = colmap_cam_to_opengl_pose(cam)
            cam_node = pr_scene.add(pr_cam, pose=cam_pose)

            color_rgba, _ = renderer.render(pr_scene, flags=flags)
            pr_scene.remove_node(cam_node)

            stem = Path(cam.image_name).stem if getattr(cam, "image_name", None) else f"cam_{idx:04d}"
            save_path = out_dir / f"{idx:04d}_{stem}.png"
            Image.fromarray(color_rgba, mode="RGBA").save(save_path)
            saved_paths.append(save_path)
    finally:
        if renderer is not None:
            renderer.delete()

    return saved_paths
