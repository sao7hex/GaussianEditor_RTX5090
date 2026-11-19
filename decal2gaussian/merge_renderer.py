import cv2
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image
from refine_renderer import refine_Image_merge


def _index_original_images(orig_images_dir: Path) -> Dict[str, Path]:
    """colmap/images/ 内のファイルを stem (拡張子なし名) -> Path で索引化する"""
    mapping: Dict[str, Path] = {}
    if not orig_images_dir.exists():
        return mapping
    for p in orig_images_dir.iterdir():
        if p.is_file():
            mapping[p.stem] = p
    return mapping


def merge_occluded_with_original_images(
    cameras: List,
    colmap_dir: str,
    occluded_dir: str,
    output_dir: str,
    cam_indices: Optional[List[int]] = None,
) -> List[Path]:
    """
    オクルーダー版レンダリング画像 (前面) と COLMAP の元画像 (背面) を
    ファイルパスから読み込んで 2D レイヤー合成し、output_dir に保存する。
    """
    orig_images_dir = Path(colmap_dir) / "images"
    orig_map = _index_original_images(orig_images_dir)

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
        bg_path = orig_map.get(stem)
        if not fg_path.exists() or bg_path is None or not bg_path.exists():
            continue

        merged_bgr = refine_Image_merge(str(bg_path), str(fg_path))
        merged_rgb = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)
        save_path = out_dir / filename
        Image.fromarray(merged_rgb).save(save_path)
        saved_paths.append(save_path)

    return saved_paths
