import os
import random
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import trange

from gaussiansplatting.arguments import OptimizationParams, PipelineParams
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.utils.loss_utils import l1_loss, ssim


def _load_merged_targets(
    cameras: List,
    merged_dir: str,
    occluded_dir: Optional[str] = None,
    dilate_px: int = 4,
) -> List[Tuple[object, torch.Tensor, Optional[torch.Tensor]]]:
    """
    renders_merged/ 内の画像と対応する Camera オブジェクト、
    および occluded_dir のアルファチャンネルから生成した 2D デカールマスクを読み込む。
    """
    m_dir = Path(merged_dir)
    if not m_dir.exists():
        raise FileNotFoundError(f"renders_merged フォルダが見つかりません: {m_dir}")

    occ_dir = Path(occluded_dir) if occluded_dir is not None else None
    targets: List[Tuple[object, torch.Tensor, Optional[torch.Tensor]]] = []

    for idx, cam in enumerate(cameras):
        stem = Path(cam.image_name).stem if getattr(cam, "image_name", None) else f"cam_{idx:04d}"
        filename = f"{idx:04d}_{stem}.png"
        img_path = m_dir / filename
        if not img_path.exists():
            continue

        target_size = (int(cam.image_width), int(cam.image_height))
        img = Image.open(img_path).convert("RGB")
        if img.size != target_size:
            img = img.resize(target_size, Image.BILINEAR)
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        gt_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().cuda()

        mask_tensor: Optional[torch.Tensor] = None
        if occ_dir is not None:
            occ_path = occ_dir / filename
            if occ_path.exists():
                occ_img = Image.open(occ_path).convert("RGBA")
                if occ_img.size != target_size:
                    occ_img = occ_img.resize(target_size, Image.NEAREST)
                alpha_np = np.asarray(occ_img, dtype=np.uint8)[:, :, 3]
                mask_t = torch.from_numpy((alpha_np > 12).astype(np.float32))[None, None, :, :].cuda()
                if dilate_px > 0 and mask_t.sum() > 0:
                    k = dilate_px * 2 + 1
                    mask_t = F.max_pool2d(mask_t, kernel_size=k, stride=1, padding=dilate_px)
                mask_tensor = mask_t.squeeze(0)  # [1, H, W]

        targets.append((cam, gt_tensor, mask_tensor))

    if not targets:
        raise FileNotFoundError(f"学習対象の合成画像が見つかりません: {m_dir}")
    return targets


@torch.no_grad()
def _compute_decal_gaussian_mask(
    gaussian,
    targets: List[Tuple[object, torch.Tensor, Optional[torch.Tensor]]],
    min_hit_ratio: float = 0.25,
) -> torch.Tensor:
    """
    各カメラの 2D デカールマスクへ全ガウシアンを投影し、
    デカール領域に関係するガウシアンだけを True にしたマスクを返す。
    """
    xyz = gaussian.get_xyz.detach()
    n_pts = xyz.shape[0]
    xyz_h = torch.cat([xyz, torch.ones((n_pts, 1), dtype=xyz.dtype, device=xyz.device)], dim=1)

    visible_count = torch.zeros(n_pts, dtype=torch.int32, device=xyz.device)
    hit_count = torch.zeros(n_pts, dtype=torch.int32, device=xyz.device)

    for cam, _, mask_2d in targets:
        if mask_2d is None or mask_2d.sum() == 0:
            continue

        H, W = mask_2d.shape[1], mask_2d.shape[2]
        clip = xyz_h @ cam.full_proj_transform
        w = clip[:, 3]
        valid_z = w > 0.01
        w_safe = w.clamp(min=1e-6)
        ndc_x = clip[:, 0] / w_safe
        ndc_y = clip[:, 1] / w_safe

        in_view = valid_z & (ndc_x >= -1.0) & (ndc_x <= 1.0) & (ndc_y >= -1.0) & (ndc_y <= 1.0)
        if not torch.any(in_view):
            continue

        visible_count[in_view] += 1
        px = ((ndc_x[in_view] + 1.0) * 0.5 * W).long().clamp(0, W - 1)
        py = ((ndc_y[in_view] + 1.0) * 0.5 * H).long().clamp(0, H - 1)
        hits = mask_2d[0, py, px] > 0.5
        hit_indices = torch.nonzero(in_view, as_tuple=False).squeeze(1)[hits]
        hit_count[hit_indices] += 1

    if hit_count.max() == 0:
        print("  [Mask] デカール領域に対応する点が検出されなかったため全点を更新対象にします。")
        return torch.ones(n_pts, dtype=torch.bool, device=xyz.device)

    ratio = hit_count.float() / visible_count.clamp(min=1).float()
    decal_mask = (hit_count >= 1) & (ratio >= min_hit_ratio)
    if not torch.any(decal_mask):
        decal_mask = hit_count >= 1

    selected = int(decal_mask.sum().item())
    print(f"  [Mask] 更新対象ガウシアンをデカール周辺に限定: {selected:,} / {n_pts:,} 点 ({selected / max(n_pts, 1) * 100:.1f}%)")
    return decal_mask


def train_3dgs_from_merged_images(
    gaussian,
    scene,
    cameras: List,
    merged_dir: str,
    output_ply_path: str,
    iterations: int = 3000,
    from_scratch: bool = False,
    occluded_dir: Optional[str] = None,
) -> Path:
    """
    renders_merged/ 内の画像を教師画像として 3DGS シーンを学習・最適化し、PLY として保存する。
    """
    use_mask = (not from_scratch) and (occluded_dir is not None)
    targets = _load_merged_targets(
        cameras,
        merged_dir,
        occluded_dir=occluded_dir if use_mask else None,
    )
    if len(targets) < 5:
        print(f"  [Warning] 学習画像が {len(targets)} 枚しかありません。全カメラをレンダリングしてから実行することを推奨します。")

    dummy_parser = argparse.ArgumentParser()
    if from_scratch:
        opt = OptimizationParams(dummy_parser, max_steps=iterations)
    else:
        opt = OptimizationParams(
            dummy_parser,
            max_steps=iterations,
            lr_scaler=0.2,
            lr_final_scaler=0.2,
            color_lr_scaler=0.3,
            opacity_lr_scaler=0.2,
            scaling_lr_scaler=0.3,
            rotation_lr_scaler=0.3,
        )
    pipe = PipelineParams(argparse.ArgumentParser())
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    cameras_extent = getattr(scene, "cameras_extent", 1.0)
    gaussian.spatial_lr_scale = cameras_extent

    n_pts = gaussian.get_xyz.shape[0]
    if use_mask:
        gaussian.mask = _compute_decal_gaussian_mask(gaussian, targets)
        decal_targets = [t for t in targets if t[2] is not None and t[2].sum() > 0]
        train_targets = decal_targets if decal_targets else targets
    else:
        gaussian.mask = torch.ones(n_pts, dtype=torch.bool, device="cuda")
        train_targets = targets

    gaussian._generation = torch.zeros(n_pts, dtype=torch.int64, device="cuda")
    gaussian.training_setup(opt)
    gaussian.apply_grad_mask(gaussian.mask)
    gaussian.update_anchor()

    if from_scratch:
        gaussian.active_sh_degree = 0
        densify_from = min(500, iterations // 6)
        densify_until = int(iterations * 0.5)
        densify_max_grad = opt.densify_grad_threshold
    else:
        gaussian.active_sh_degree = gaussian.max_sh_degree
        densify_from = min(200, iterations // 10)
        densify_until = int(iterations * 0.4)
        densify_max_grad = opt.densify_grad_threshold * 2.0

    print(f"\n[3DGS Training] 画像数={len(train_targets)} 枚, ステップ数={iterations}, モード={'新規学習' if from_scratch else '既存PLYから追加学習'}")

    pbar = trange(1, iterations + 1, desc="Training 3DGS")
    for step in pbar:
        gaussian.update_learning_rate(step)
        if from_scratch and step % 1000 == 0:
            gaussian.oneupSHdegree()

        cam, gt_image, mask_2d = random.choice(train_targets)
        render_pkg = render(cam, gaussian, pipe, bg_color)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if mask_2d is not None:
            # デカール領域外の背景ピクセルは勾配を発生させない
            eval_image = image * mask_2d + gt_image.detach() * (1.0 - mask_2d)
        else:
            eval_image = image

        Ll1 = l1_loss(eval_image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(eval_image, gt_image))
        loss.backward()

        with torch.no_grad():
            if step < densify_until and viewspace_point_tensor.grad is not None:
                gaussian.max_radii2D[visibility_filter] = torch.max(
                    gaussian.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussian.add_densification_stats(viewspace_point_tensor.grad, visibility_filter)

                if step >= densify_from and step % opt.densification_interval == 0:
                    gaussian.densify_and_prune(
                        max_grad=densify_max_grad,
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
