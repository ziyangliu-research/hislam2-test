import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from gaussian.renderer import render
from gaussian.utils.camera_utils import Camera
from gaussian.utils.loss_utils import psnr, ssim


@torch.no_grad()
def eval_rendering_subset(
    images,
    traj_w2c,
    indices,
    gaussians,
    background,
    projection_matrix,
    K,
    output_json=None,
):
    """Evaluate PSNR/SSIM/LPIPS for an explicit frame subset without changing the map.

    The LPIPS definition intentionally matches HI-SLAM2's original evaluator:
    AlexNet backbone with ``normalize=True`` for RGB tensors in [0, 1].

    Args:
        images: dict mapping integer timestamps to HI-SLAM2 image tensors.
        traj_w2c: per-frame 4x4 world-to-camera matrices.
        indices: iterable of integer frame timestamps to evaluate.
    """
    psnr_values = []
    ssim_values = []
    lpips_values = []
    valid_indices = []

    lpips_metric = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    lpips_metric.eval()

    for idx in indices:
        idx = int(idx)
        if idx not in images or idx >= len(traj_w2c):
            continue

        image = images[idx]
        frame = Camera.init_from_tracking(
            image.squeeze() / 255.0,
            None,
            None,
            traj_w2c[idx],
            idx,
            projection_matrix,
            K,
        )
        gtimage = frame.original_image.cuda()
        rendering = render(frame, gaussians, background)
        pred = torch.clamp(rendering["render"], 0.0, 1.0)

        mask = gtimage > 0
        psnr_score = psnr(pred[mask].unsqueeze(0), gtimage[mask].unsqueeze(0))
        ssim_score = ssim(pred.unsqueeze(0), gtimage.unsqueeze(0))
        lpips_score = lpips_metric(pred.unsqueeze(0), gtimage.unsqueeze(0))

        psnr_values.append(float(psnr_score.item()))
        ssim_values.append(float(ssim_score.item()))
        lpips_values.append(float(lpips_score.item()))
        valid_indices.append(idx)

    result = {
        "count": len(valid_indices),
        "mean_psnr": float(np.mean(psnr_values)) if psnr_values else float("nan"),
        "mean_ssim": float(np.mean(ssim_values)) if ssim_values else float("nan"),
        "mean_lpips": float(np.mean(lpips_values)) if lpips_values else float("nan"),
        "indices": valid_indices,
    }

    if output_json is not None:
        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result
