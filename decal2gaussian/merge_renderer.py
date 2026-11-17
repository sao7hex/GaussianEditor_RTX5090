from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image


def _camera_original_to_pil(cam) -> Image.Image:
    """COLMAP Camera の original_image ([3, H, W], 0..1) を RGBA PIL Image に変換する"""
    img_np = (
        cam.original_image.clamp(0.0, 1.0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
        * 255.0
    ).round().astype(np.uint8)
    return Image.fromarray(img_np, mode="RGB").convert("RGBA")


def merge_occluded_with_original_images(
    cameras: List,
    occluded_dir: str,
    output_dir: str,
    cam_indices: Optional[List[int]] = None,
) -> List[Path]:
    """
    オクルーダー版レンダリング画像 (前面) と COLMAP の元画像 (背面) を
    2D レイヤーとしてアルファ合成し、output_dir に保存する。
    """
    occ_dir = Path(occluded_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_indices = cam_indices if cam_indices is not None else list(range(len(cameras)))
    saved_paths: List[Path] = []

    for idx in target_indices:
        if idx < 0 or idx >= len(cameras):
            continue

        cam = cameras[idx]
        stem = Path(cam.image_name).stem if getattr(cam, "image_name", None) else f"cam_{idx:04d}"
        filename = f"{idx:04d}_{stem}.png"
        fg_path = occ_dir / filename
        if not fg_path.exists():
            continue

        bg_rgba = _camera_original_to_pil(cam)
        fg_rgba = Image.open(fg_path).convert("RGBA")
        if fg_rgba.size != bg_rgba.size:
            fg_rgba = fg_rgba.resize(bg_rgba.size, Image.BILINEAR)

        merged = Image.alpha_composite(bg_rgba, fg_rgba).convert("RGB")
        save_path = out_dir / filename
        merged.save(save_path)
        saved_paths.append(save_path)

    return saved_paths
