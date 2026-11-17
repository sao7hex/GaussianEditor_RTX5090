import os
import random
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import trange

from gaussiansplatting.arguments import OptimizationParams, PipelineParams
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.utils.loss_utils import l1_loss, ssim


def _load_merged_targets(cameras: List, merged_dir: str) -> List[Tuple[object, torch.Tensor]]:
    """renders_merged/ 内の画像と対応する Camera オブジェクトのペアを読み込む"""
    m_dir = Path(merged_dir)
    if not m_dir.exists():
        raise FileNotFoundError(f"renders_merged フォルダが見つかりません: {m_dir}")

    targets: List[Tuple[object, torch.Tensor]] = []
    for idx, cam in enumerate(cameras):
        stem = Path(cam.image_name).stem if getattr(cam, "image_name", None) else f"cam_{idx:04d}"
        img_path = m_dir / f"{idx:04d}_{stem}.png"
        if not img_path.exists():
            continue

        img_np = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.float32) / 255.0
        gt_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().cuda()
        targets.append((cam, gt_tensor))

    if not targets:
        raise FileNotFoundError(f"学習対象の合成画像が見つかりません: {m_dir}")
    return targets


def train_3dgs_from_merged_images(
    gaussian,
    scene,
    cameras: List,
    merged_dir: str,
    output_ply_path: str,
    iterations: int = 3000,
    from_scratch: bool = False,
) -> Path:
    """
    renders_merged/ 内の画像を教師画像として 3DGS シーンを学習・最適化し、PLY として保存する。
    """
    targets = _load_merged_targets(cameras, merged_dir)
    if len(targets) < 5:
        print(f"  [Warning] 学習画像が {len(targets)} 枚しかありません。全カメラをレンダリングしてから実行することを推奨します。")

    dummy_parser = argparse.ArgumentParser()
    opt = OptimizationParams(dummy_parser, max_steps=iterations)
    pipe = PipelineParams(argparse.ArgumentParser())
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    cameras_extent = getattr(scene, "cameras_extent", 1.0)
    gaussian.spatial_lr_scale = cameras_extent

    n_pts = gaussian.get_xyz.shape[0]
    gaussian.mask = torch.ones(n_pts, dtype=torch.bool, device="cuda")
    gaussian._generation = torch.zeros(n_pts, dtype=torch.int64, device="cuda")
    gaussian.training_setup(opt)

    if from_scratch:
        gaussian.active_sh_degree = 0
        densify_from = min(500, iterations // 6)
        densify_until = int(iterations * 0.5)
    else:
        gaussian.active_sh_degree = gaussian.max_sh_degree
        densify_from = min(200, iterations // 10)
        densify_until = int(iterations * 0.7)

    print(f"\n[3DGS Training] 画像数={len(targets)} 枚, ステップ数={iterations}, モード={'新規学習' if from_scratch else '既存PLYから追加学習'}")

    pbar = trange(1, iterations + 1, desc="Training 3DGS")
    for step in pbar:
        gaussian.update_learning_rate(step)
        if from_scratch and step % 1000 == 0:
            gaussian.oneupSHdegree()

        cam, gt_image = random.choice(targets)
        render_pkg = render(cam, gaussian, pipe, bg_color)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        with torch.no_grad():
            if step < densify_until and viewspace_point_tensor.grad is not None:
                gaussian.max_radii2D[visibility_filter] = torch.max(
                    gaussian.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussian.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if step >= densify_from and step % opt.densification_interval == 0:
                    gaussian.densify_and_prune(
                        max_grad=opt.densify_grad_threshold,
                        max_densify_percent=0.01,
                        min_opacity=0.005,
                        extent=cameras_extent,
                        max_screen_size=20,
                    )

                if from_scratch and step % opt.opacity_reset_interval == 0:
                    gaussian.reset_opacity()

        gaussian.optimizer.step()
        gaussian.optimizer.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == iterations:
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Pts": f"{gaussian.get_xyz.shape[0]:,}",
            })

    out_path = Path(output_ply_path)
    os.makedirs(out_path.parent, exist_ok=True)
    gaussian.save_ply(str(out_path))
    return out_path
