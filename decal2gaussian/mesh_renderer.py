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


def _split_submeshes_by_material(geom: trimesh.Trimesh, node_name: str) -> List[Tuple[str, trimesh.Trimesh]]:
    """マルチマテリアルを持つメッシュをマテリアル単位のサブメッシュに分割する"""
    visual = getattr(geom, "visual", None)
    face_materials = getattr(visual, "face_materials", None) if visual is not None else None
    if face_materials is None or len(face_materials) != len(geom.faces):
        return [(node_name, geom)]

    unique_mats = np.unique(face_materials)
    if len(unique_mats) <= 1:
        return [(node_name, geom)]

    materials = getattr(getattr(visual, "material", None), "materials", None)
    submeshes: List[Tuple[str, trimesh.Trimesh]] = []
    for mat_idx in unique_mats:
        face_indices = np.where(face_materials == mat_idx)[0]
        if len(face_indices) == 0:
            continue
        sub = geom.submesh([face_indices], append=False)
        sub_mesh = sub[0] if isinstance(sub, list) and len(sub) > 0 else sub
        if materials is not None and 0 <= mat_idx < len(materials):
            sub_mesh.visual.material = materials[mat_idx]
        if len(sub_mesh.vertices) > 0 and len(sub_mesh.faces) > 0:
            submeshes.append((f"{node_name}_mat{mat_idx}", sub_mesh))

    return submeshes if submeshes else [(node_name, geom)]


def _prepare_trimesh_for_pyrender(
    geom: trimesh.Trimesh,
    alpha_mode: str = "BLEND",
    alpha_cutoff: float = 0.05,
) -> Tuple[trimesh.Trimesh, bool]:
    """
    pyrender.Mesh.from_trimesh へ渡す前に、trimesh 側のマテリアルとテクスチャを
    透過対応 (RGBA + PBRMaterial) に正規化する。
    """
    geom = geom.copy()
    visual = getattr(geom, "visual", None)
    mat = getattr(visual, "material", None) if visual is not None else None
    if mat is None:
        return geom, False

    if not isinstance(mat, trimesh.visual.material.PBRMaterial):
        to_pbr = getattr(mat, "to_pbr", None)
        if callable(to_pbr):
            mat = to_pbr()
        else:
            tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            mat = trimesh.visual.material.PBRMaterial(
                baseColorTexture=tex,
                baseColorFactor=getattr(mat, "diffuse", [255, 255, 255, 255]),
            )

    has_alpha = False
    base_factor = getattr(mat, "baseColorFactor", None)
    if base_factor is not None and len(base_factor) >= 4:
        factor_arr = np.asarray(base_factor, dtype=np.float32)
        alpha_val = factor_arr[3] / 255.0 if factor_arr.max() > 1.0 else factor_arr[3]
        if alpha_val < 1.0:
            has_alpha = True

    tex = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
    if tex is not None:
        if not isinstance(tex, Image.Image):
            tex = Image.fromarray(np.asarray(tex))
        if tex.mode != "RGBA":
            tex = tex.convert("RGBA")
        tex_arr = np.asarray(tex)
        if int(tex_arr[:, :, 3].min()) < 255:
            has_alpha = True
        mat.baseColorTexture = tex

    current_mode = getattr(mat, "alphaMode", None)
    if has_alpha or current_mode in ("MASK", "BLEND"):
        mat.alphaMode = alpha_mode
        if alpha_mode == "MASK":
            mat.alphaCutoff = float(alpha_cutoff)
        has_alpha = True

    geom.visual.material = mat
    return geom, has_alpha


def _create_pyrender_scene(flat: bool) -> "pyrender.Scene":
    import pyrender

    scene = pyrender.Scene(
        bg_color=[0.0, 0.0, 0.0, 0.0],
        ambient_light=[1.0, 1.0, 1.0],
    )
    if not flat:
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=np.eye(4))
    return scene


def _build_pyrender_mesh(
    geom: trimesh.Trimesh,
    has_alpha: bool,
    alpha_mode: str,
    alpha_cutoff: float,
):
    import pyrender

    pr_mesh = pyrender.Mesh.from_trimesh(geom, smooth=False)
    for prim in pr_mesh.primitives:
        if prim.material is not None and has_alpha:
            prim.material.alphaMode = alpha_mode
            if alpha_mode == "MASK":
                prim.material.alphaCutoff = float(alpha_cutoff)
            tex = getattr(prim.material, "baseColorTexture", None)
            if tex is not None:
                tex.source_channels = "RGBA"
    return pr_mesh


def render_glb_from_cameras(
    meshes: List[Tuple[str, trimesh.Trimesh]],
    cameras: List,
    output_dir: str,
    cam_indices: Optional[List[int]] = None,
    flat: bool = True,
    alpha_mode: str = "BLEND",
    alpha_cutoff: float = 0.05,
    depth_tolerance: float = 0.0005,
) -> List[Path]:
    """
    読み込んだ GLB メッシュ群を COLMAP カメラ視点からオフスクリーンレンダリングし、RGBA PNG として保存する。
    透明ありメッシュは個別に描画して深度バッファ合成することで、透明領域が奥のメッシュを消す問題や
    同一平面上の Z-fighting を防ぐ。
    """
    import pyrender

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    opaque_scene = _create_pyrender_scene(flat=flat)
    has_opaque = False
    transparent_scenes: List[Tuple[str, "pyrender.Scene"]] = []

    for node_name, geom in meshes:
        if len(geom.vertices) == 0 or len(geom.faces) == 0:
            continue
        for sub_name, sub_geom in _split_submeshes_by_material(geom, node_name):
            try:
                norm_geom, has_alpha = _prepare_trimesh_for_pyrender(
                    sub_geom,
                    alpha_mode=alpha_mode,
                    alpha_cutoff=alpha_cutoff,
                )
                pr_mesh = _build_pyrender_mesh(norm_geom, has_alpha, alpha_mode, alpha_cutoff)
                if has_alpha:
                    t_scene = _create_pyrender_scene(flat=flat)
                    t_scene.add(pr_mesh, name=sub_name)
                    transparent_scenes.append((sub_name, t_scene))
                else:
                    opaque_scene.add(pr_mesh, name=sub_name)
                    has_opaque = True
                print(f"  [Mesh] '{sub_name}': faces={len(norm_geom.faces):,}, transparent={has_alpha}")
            except Exception as e:
                print(f"  [Warning] メッシュ '{sub_name}' の pyrender 変換をスキップしました: {e}")

    if not has_opaque and not transparent_scenes:
        raise RuntimeError("レンダリング可能な有効メッシュがありません。")

    target_indices = cam_indices if cam_indices is not None else list(range(len(cameras)))
    flags = pyrender.RenderFlags.RGBA
    if flat:
        flags |= pyrender.RenderFlags.FLAT

    renderer = None
    current_size = None
    saved_paths: List[Path] = []
    alpha_thresh_u8 = max(1, int(round(alpha_cutoff * 255)))

    try:
        for i, idx in enumerate(target_indices):
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

            # 1. 不透明メッシュをベースとして描画
            if has_opaque:
                cam_node = opaque_scene.add(pr_cam, pose=cam_pose)
                comp_rgba, comp_depth = renderer.render(opaque_scene, flags=flags)
                opaque_scene.remove_node(cam_node)
                comp_rgba = comp_rgba.astype(np.float32)
                z_buffer = np.where(comp_depth > 0.0, comp_depth, np.inf).astype(np.float32)
            else:
                comp_rgba = np.zeros((H, W, 4), dtype=np.float32)
                z_buffer = np.full((H, W), np.inf, dtype=np.float32)

            # 2. 透明ありメッシュ (デカール等) を1つずつ描画し、有効アルファ部分のみ Z テストして合成
            for _, t_scene in transparent_scenes:
                cam_node = t_scene.add(pr_cam, pose=cam_pose)
                layer_rgba, layer_depth = renderer.render(t_scene, flags=flags)
                t_scene.remove_node(cam_node)

                valid_pixel = (layer_rgba[:, :, 3] >= alpha_thresh_u8) & (layer_depth > 0.0)
                is_closer = layer_depth < (z_buffer - depth_tolerance)
                is_coplanar = np.abs(layer_depth - z_buffer) <= depth_tolerance
                update_mask = valid_pixel & (is_closer | is_coplanar)

                if np.any(update_mask):
                    src_rgb = layer_rgba[update_mask, :3].astype(np.float32)
                    src_a = layer_rgba[update_mask, 3:4].astype(np.float32) / 255.0
                    dst_rgb = comp_rgba[update_mask, :3]
                    dst_a = comp_rgba[update_mask, 3:4] / 255.0

                    out_a = src_a + dst_a * (1.0 - src_a)
                    comp_rgba[update_mask, :3] = src_rgb * src_a + dst_rgb * (1.0 - src_a)
                    comp_rgba[update_mask, 3:4] = np.clip(out_a * 255.0, 0.0, 255.0)
                    z_buffer[update_mask] = layer_depth[update_mask]

            out_u8 = np.clip(comp_rgba, 0.0, 255.0).astype(np.uint8)
            if i == 0:
                a_min, a_max = int(out_u8[:, :, 3].min()), int(out_u8[:, :, 3].max())
                print(f"  [Render Check] cam={idx}, output shape={out_u8.shape}, alpha range=[{a_min}, {a_max}]")

            stem = Path(cam.image_name).stem if getattr(cam, "image_name", None) else f"cam_{idx:04d}"
            save_path = out_dir / f"{idx:04d}_{stem}.png"
            Image.fromarray(out_u8, mode="RGBA").save(save_path)
            saved_paths.append(save_path)
    finally:
        if renderer is not None:
            renderer.delete()

    return saved_paths
