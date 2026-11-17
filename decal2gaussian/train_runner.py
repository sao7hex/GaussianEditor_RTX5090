import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from PIL import Image


def _index_original_images(orig_images_dir: Path) -> Dict[str, Path]:
    """colmap/images/ 内のファイルを stem (拡張子なし名) -> Path で索引化する"""
    mapping: Dict[str, Path] = {}
    if not orig_images_dir.exists():
        return mapping
    for p in orig_images_dir.iterdir():
        if p.is_file():
            mapping[p.stem] = p
    return mapping


def prepare_colmap_dataset_from_merged(
    colmap_dir: str,
    merged_dir: str,
    cameras: List,
    work_dataset_dir: str,
) -> Path:
    """
    gaussiansplatting/train.py がそのまま読み込める形式で
    sparse/ と images/ (renders_merged で差し替え済み) を構築する。
    """
    src_colmap = Path(colmap_dir)
    src_sparse = src_colmap / "sparse"
    src_images = src_colmap / "images"
    m_dir = Path(merged_dir)
    dst_dataset = Path(work_dataset_dir)
    dst_sparse = dst_dataset / "sparse"
    dst_images = dst_dataset / "images"

    if not src_sparse.exists():
        raise FileNotFoundError(f"sparse フォルダが見つかりません: {src_sparse}")

    if dst_sparse.exists():
        shutil.rmtree(dst_sparse)
    shutil.copytree(src_sparse, dst_sparse)

    if dst_images.exists():
        shutil.rmtree(dst_images)
    dst_images.mkdir(parents=True, exist_ok=True)

    orig_map = _index_original_images(src_images)
    replaced_count = 0
    copied_orig_count = 0

    for idx, cam in enumerate(cameras):
        stem = Path(cam.image_name).stem if getattr(cam, "image_name", None) else f"cam_{idx:04d}"
        merged_path = m_dir / f"{idx:04d}_{stem}.png"
        orig_path = orig_map.get(stem)

        if orig_path is not None:
            dst_name = orig_path.name
        else:
            dst_name = f"{stem}.png"

        dst_path = dst_images / dst_name
        if merged_path.exists():
            # train.py が参照する元の拡張子に合わせて保存
            if dst_path.suffix.lower() in (".jpg", ".jpeg"):
                Image.open(merged_path).convert("RGB").save(dst_path, quality=95)
            else:
                Image.open(merged_path).convert("RGB").save(dst_path)
            replaced_count += 1
        elif orig_path is not None and orig_path.exists():
            shutil.copy2(orig_path, dst_path)
            copied_orig_count += 1

    print(f"  [Dataset] train.py 用データセットを構築しました: {dst_dataset}")
    print(f"    - 合成画像で差し替え: {replaced_count} 枚, 元画像をそのまま使用: {copied_orig_count} 枚")
    return dst_dataset


def run_gaussiansplatting_train_py(
    root_dir: Path,
    colmap_dir: str,
    merged_dir: str,
    cameras: List,
    work_dataset_dir: str,
    model_out_dir: str,
    iterations: int = 30000,
) -> Path:
    """
    renders_merged/ の画像で仮COLMAPデータセットを構築し、
    gaussiansplatting/train.py をサブプロセスで実行して 3DGS を学習する。
    """
    dataset_dir = prepare_colmap_dataset_from_merged(
        colmap_dir=colmap_dir,
        merged_dir=merged_dir,
        cameras=cameras,
        work_dataset_dir=work_dataset_dir,
    )

    gs_dir = Path(root_dir) / "gaussiansplatting"
    train_script = gs_dir / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"train.py が見つかりません: {train_script}")

    out_dir = Path(model_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "train.py",
        "-s",
        str(dataset_dir.resolve()),
        "-m",
        str(out_dir.resolve()),
        "--iterations",
        str(iterations),
        "--quiet",
    ]
    print(f"\n[train.py 実行] cwd={gs_dir}")
    print("  cmd:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(gs_dir), check=True)

    result_ply = out_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    return result_ply
