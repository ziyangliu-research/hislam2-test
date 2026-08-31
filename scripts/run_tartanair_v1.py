#!/usr/bin/env python3
"""Run HI-SLAM2 on TartanAir v1 Stereo Challenge sequences as monocular input.

HI-SLAM2 is a monocular system. By default this runner feeds only image_left
into demo.py. The right image directory is supported only as an alternative
monocular stream; stereo fusion is intentionally not added here.

Ground-truth files are validated and printed for reproducibility, but are not
used by HI-SLAM2 during reconstruction. Trajectory evaluation is intentionally
kept separate from this first dataset-adaptation step.
"""

import argparse
import os
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = Path("/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/stereo")
DEFAULT_GT_ROOT = Path("/home/shiyo/Desktop/Datasets/TartanAir_Stereo_Challenge/ground_truth/stereo_gt")
SEQUENCES = [f"SE{i:03d}" for i in range(8)] + [f"SH{i:03d}" for i in range(8)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run original HI-SLAM2 on TartanAir v1 challenge images."
    )
    parser.add_argument(
        "sequence",
        nargs="?",
        default="SE000",
        choices=SEQUENCES,
        help="sequence to run (default: SE000)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run all SE000-SE007 and SH000-SH007 sequences",
    )
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

    output_dir = (
        args.output_root
        / f"{sequence}_{args.camera}_start{args.start}_len{run_length}"
    )

    cmd = [
        sys.executable,
        str(REPO_ROOT / "demo.py"),
        "--imagedir",
        str(image_dir),
        "--calib",
        str(args.calib),
        "--config",
        str(args.config),
        "--output",
        str(output_dir),
        "--start",
        str(args.start),
        "--length",
        str(run_length),
    ]
    if args.buffer is not None:
        cmd.extend(["--buffer", str(args.buffer)])

    print("=" * 80)
    print(f"Sequence : {sequence}")
    print(f"Camera   : {args.camera} (monocular input)")
    print(f"Images   : {image_dir}")
    print(f"GT pose  : {gt_path} (not used during reconstruction)")
    print(f"Frames   : start={args.start}, length={run_length}, total={len(image_files)}")
    print(f"Output   : {output_dir}")
    print("Command  :", " ".join(cmd))

    if args.dry_run:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # HI-SLAM2 internally uses cuda:0. CUDA_VISIBLE_DEVICES remaps the selected
    # physical GPU to logical cuda:0 without changing the original code.
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def main():
    args = parse_args()
    sequences = SEQUENCES if args.all else [args.sequence]
    for sequence in sequences:
        run_sequence(args, sequence)


if __name__ == "__main__":
    main()
