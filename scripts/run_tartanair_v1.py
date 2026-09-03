#!/usr/bin/env python3
"""Run HI-SLAM2 on TartanAir v1 Stereo Challenge sequences as monocular input.

Benchmark mode implements a deterministic 8:2 holdout protocol (default
holdout_5_4): every fifth frame with local index % 5 == 4 is pose-only and is
never allowed to enter the Gaussian map. A single HI-SLAM2 run produces both
an online/pre-global snapshot and the original final globally refined result.
"""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo")
DEFAULT_GT_ROOT = Path("/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt")
SEQUENCES = [f"SE{i:03d}" for i in range(8)] + [f"SH{i:03d}" for i in range(8)]
COMPARISON4 = ["SH000", "SH001", "SH002", "SH003"]
COMPARISON8 = [
    "SE000", "SE001", "SE002", "SE003",
    "SH000", "SH001", "SH002", "SH003",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run HI-SLAM2 on TartanAir v1 challenge images."
    )
    parser.add_argument(
        "sequence",
        nargs="?",
        default="SE000",
        choices=SEQUENCES,
        help="sequence to run (default: SE000)",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="run all SE000-SE007 and SH000-SH007 sequences",
    )
    group.add_argument(
        "--comparison4",
        action="store_true",
        help="run SH000-SH003",
    )
    group.add_argument(
        "--comparison8",
        action="store_true",
        help="run SE000-SE003 and SH000-SH003",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="enable 8:2 holdout plus online/pre-global and official-final evaluation",
    )
    parser.add_argument("--holdout-every", type=int, default=5)
    parser.add_argument("--holdout-offset", type=int, default=4)
    parser.add_argument(
        "--camera",
        choices=["left", "right"],
        default="left",
        help="monocular image stream to use (default: left)",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--gt-root", type=Path, default=DEFAULT_GT_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "tartanair_v1",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=REPO_ROOT / "calib" / "tartanair_v1.txt",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "config" / "tartanair_v1_config.yaml",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--length",
        type=int,
        default=None,
        help="number of frames to process; omit to process all remaining frames",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=None,
        help="optional HI-SLAM2 keyframe buffer override",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate paths and print commands without running HI-SLAM2",
    )
    return parser.parse_args()


def umeyama_align(est, gt, with_scale):
    """Align estimated xyz to GT with SE(3), optionally Sim(3)."""
    est = np.asarray(est, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mu_est = est.mean(axis=0)
    mu_gt = gt.mean(axis=0)
    est_c = est - mu_est
    gt_c = gt - mu_gt

    cov = (gt_c.T @ est_c) / len(est)
    U, singular, Vt = np.linalg.svd(cov)
    sign = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        sign[-1] = -1.0
    S = np.diag(sign)
    R = U @ S @ Vt

    if with_scale:
        var_est = np.sum(est_c * est_c) / len(est)
        scale = float(np.sum(singular * sign) / var_est)
    else:
        scale = 1.0

    t = mu_gt - scale * (R @ mu_est)
    aligned = (scale * (R @ est.T)).T + t
    return aligned, scale, R, t


def evaluate_trajectory(traj_path, gt_path, expected_frames):
    traj = np.loadtxt(traj_path)
    gt = np.loadtxt(gt_path)
    traj = np.atleast_2d(traj)
    gt = np.atleast_2d(gt)

    finite = np.isfinite(traj).all(axis=1)
    traj = traj[finite]
    frame_ids = np.rint(traj[:, 0]).astype(int)
    valid = (frame_ids >= 0) & (frame_ids < len(gt))
    traj = traj[valid]
    frame_ids = frame_ids[valid]

    if len(traj) < 3:
        raise RuntimeError(f"Not enough valid poses in {traj_path}")

    est_xyz = traj[:, 1:4]
    gt_xyz = gt[frame_ids, :3]

    est_se3, _, _, _ = umeyama_align(est_xyz, gt_xyz, with_scale=False)
    ate_se3 = float(np.sqrt(np.mean(np.sum((est_se3 - gt_xyz) ** 2, axis=1))))

    est_sim3, sim3_scale, _, _ = umeyama_align(est_xyz, gt_xyz, with_scale=True)
    ate_sim3 = float(np.sqrt(np.mean(np.sum((est_sim3 - gt_xyz) ** 2, axis=1))))

    return {
        "valid_poses": int(len(traj)),
        "expected_frames": int(expected_frames),
        "maxmap_percent": float(100.0 * len(traj) / expected_frames),
        "ate_se3_m": ate_se3,
        "ate_sim3_m_aux": ate_sim3,
        "sim3_scale_aux": sim3_scale,
    }


def merged_stage_row(sequence, stage_name, render_stage, traj_eval):
    return {
        "Stage": stage_name,
        "Sequence": sequence,
        "MaxMap": traj_eval["maxmap_percent"],
        "Train_PSNR": render_stage["train"]["mean_psnr"],
        "Train_SSIM": render_stage["train"]["mean_ssim"],
        "Train_LPIPS": render_stage["train"]["mean_lpips"],
        "Test_PSNR": render_stage["test"]["mean_psnr"],
        "Test_SSIM": render_stage["test"]["mean_ssim"],
        "Test_LPIPS": render_stage["test"]["mean_lpips"],
        "ATE_m": traj_eval["ate_se3_m"],
        "ATE_Sim3_m_aux": traj_eval["ate_sim3_m_aux"],
        "FPS": render_stage.get("fps"),
        "EffectiveFPS_aux": render_stage.get("effective_fps_aux"),
        "Gaussians": render_stage["gaussians"],
        "OnlineTime_s": render_stage["online_seconds"],
        "OfflineTime_s": render_stage["offline_seconds"],
        "TotalTime_s": render_stage["total_seconds"],
    }


def run_sequence(args, sequence):
    image_dir = args.dataset_root / sequence / f"image_{args.camera}"
    gt_path = args.gt_root / f"{sequence}.txt"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"ground-truth file not found: {gt_path}")
    if not args.calib.is_file():
        raise FileNotFoundError(f"calibration file not found: {args.calib}")
    if not args.config.is_file():
        raise FileNotFoundError(f"config file not found: {args.config}")

    image_files = sorted(image_dir.glob("*.png"))
    if not image_files:
        raise RuntimeError(f"no PNG images found in: {image_dir}")
    if args.start < 0 or args.start >= len(image_files):
        raise ValueError(
            f"--start {args.start} is outside sequence with {len(image_files)} images"
        )

    remaining = len(image_files) - args.start
    run_length = remaining if args.length is None else min(args.length, remaining)
    if run_length <= 0:
        raise ValueError("--length must be positive")

    run_tag = f"{sequence}_{args.camera}_start{args.start}_len{run_length}"
    if args.benchmark:
        run_tag += f"_holdout_{args.holdout_every}_{args.holdout_offset}_dual"
    output_dir = args.output_root / run_tag

    cmd = [
        sys.executable,
        str(REPO_ROOT / "demo.py"),
        "--imagedir", str(image_dir),
        "--calib", str(args.calib),
        "--config", str(args.config),
        "--output", str(output_dir),
        "--start", str(args.start),
        "--length", str(run_length),
    ]
    if args.buffer is not None:
        cmd.extend(["--buffer", str(args.buffer)])
    if args.benchmark:
        cmd.extend([
            "--holdout-every", str(args.holdout_every),
            "--holdout-offset", str(args.holdout_offset),
            "--dual-stage-eval",
        ])

    print("=" * 100)
    print(f"Sequence : {sequence}")
    print(f"Camera   : {args.camera} (monocular input)")
    print(f"Images   : {image_dir}")
    print(f"GT pose  : {gt_path} (evaluation only; never used by reconstruction)")
    print(f"Frames   : start={args.start}, length={run_length}, total={len(image_files)}")
    if args.benchmark:
        print(f"Split    : holdout_{args.holdout_every}_{args.holdout_offset} (test frames are pose-only)")
        print("Stages   : online/pre-global + official final/global")
        print("Metrics  : PSNR / SSIM / LPIPS")
        print("Timing   : online / offline / total; metric rendering excluded")
    print(f"Output   : {output_dir}")
    print("Command  :", " ".join(cmd))

    if args.dry_run:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)

    if not args.benchmark:
        return None

    benchmark_path = output_dir / "benchmark_metrics.json"
    if not benchmark_path.is_file():
        raise RuntimeError(f"Benchmark metrics missing: {benchmark_path}")
    benchmark = json.loads(benchmark_path.read_text())

    online_traj = evaluate_trajectory(output_dir / "traj_online.txt", gt_path, run_length)
    final_traj = evaluate_trajectory(output_dir / "traj_full.txt", gt_path, run_length)

    merged = {
        "sequence": sequence,
        "split": {
            "holdout_every": args.holdout_every,
            "holdout_offset": args.holdout_offset,
            "train_count": benchmark["train_count"],
            "test_count": benchmark["test_count"],
        },
        "timing_definition": benchmark.get("timing_definition", {}),
        "online": {**benchmark["online"], "trajectory": online_traj},
        "final": {**benchmark["final"], "trajectory": final_traj},
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(merged, indent=2))

    return (
        merged_stage_row(sequence, "online", benchmark["online"], online_traj),
        merged_stage_row(sequence, "final", benchmark["final"], final_traj),
    )


def fps_text(value):
    return "-" if value is None else f"{value:.4f}"


def print_table(title, rows):
    if not rows:
        return
    print("\n" + title)
    print("=" * 188)
    header = (
        f"{'Sequence':<9} {'MaxMap':>8}  {'Train P/S/L':>28}  "
        f"{'Test P/S/L':>28}  {'ATE(m)':>10}  {'FPS':>9}  {'Gaussians':>11}  "
        f"{'Online(s)':>10}  {'Offline(s)':>11}  {'Total(s)':>10}"
    )
    print(header)
    print("-" * 188)
    for r in rows:
        train = f"{r['Train_PSNR']:.4f}/{r['Train_SSIM']:.6f}/{r['Train_LPIPS']:.6f}"
        test = f"{r['Test_PSNR']:.4f}/{r['Test_SSIM']:.6f}/{r['Test_LPIPS']:.6f}"
        print(
            f"{r['Sequence']:<9} {r['MaxMap']:>7.2f}%  {train:>28}  "
            f"{test:>28}  {r['ATE_m']:>10.6f}  {fps_text(r['FPS']):>9}  "
            f"{int(r['Gaussians']):>11d}  {r['OnlineTime_s']:>10.2f}  "
            f"{r['OfflineTime_s']:>11.2f}  {r['TotalTime_s']:>10.2f}"
        )


def save_csv(path, rows):
    if not rows:
        return
    fields = [
        "Sequence", "MaxMap",
        "Train_PSNR", "Train_SSIM", "Train_LPIPS",
        "Test_PSNR", "Test_SSIM", "Test_LPIPS",
        "ATE_m", "FPS", "Gaussians",
        "OnlineTime_s", "OfflineTime_s", "TotalTime_s",
        "ATE_Sim3_m_aux", "EffectiveFPS_aux", "Stage",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})


def main():
    args = parse_args()
    if args.all:
        sequences = SEQUENCES
    elif args.comparison8:
        sequences = COMPARISON8
    elif args.comparison4:
        sequences = COMPARISON4
    else:
        sequences = [args.sequence]

    online_rows = []
    final_rows = []
    for sequence in sequences:
        result = run_sequence(args, sequence)
        if result is not None:
            online_row, final_row = result
            online_rows.append(online_row)
            final_rows.append(final_row)

            # Show accumulated results after every completed sequence so a long
            # batch still leaves a useful visible record.
            print_table("ONLINE / PRE-GLOBAL", online_rows)
            print_table("OFFICIAL FINAL / GLOBAL REFINEMENT", final_rows)

    if args.benchmark and not args.dry_run:
        split_tag = f"holdout_{args.holdout_every}_{args.holdout_offset}"
        save_csv(args.output_root / f"summary_online_{split_tag}.csv", online_rows)
        save_csv(args.output_root / f"summary_final_{split_tag}.csv", final_rows)
        save_csv(args.output_root / f"summary_all_{split_tag}.csv", online_rows + final_rows)

        print_table("FINAL SUMMARY — ONLINE / PRE-GLOBAL", online_rows)
        print_table("FINAL SUMMARY — OFFICIAL FINAL / GLOBAL REFINEMENT", final_rows)
        print(f"\nSaved summaries under: {args.output_root}")
        print("Main ATE(m): SE(3) rigid alignment, no scale correction.")
        print("Main FPS: online stage only; final/offline stage is shown as '-'.")
        print("Online/Offline/Total times exclude PSNR/SSIM/LPIPS rendering/evaluation.")
        print("Auxiliary Sim(3) ATE and cumulative effective FPS are kept in JSON/CSV only.")


if __name__ == "__main__":
    main()
