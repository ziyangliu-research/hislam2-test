import os    # nopep8
import sys   # nopep8
sys.path.append(os.path.join(os.path.dirname(__file__), 'hislam2'))   # nopep8
import time
import json
import torch
import cv2
import re
import os
import argparse
import warnings
import numpy as np
import lietorch
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (100000, rlimit[1]))

# HI-SLAM2 was written against the torch.cuda.amp.autocast API used by
# PyTorch 2.1. Newer PyTorch versions keep the same behavior but emit a
# FutureWarning recommending torch.amp.autocast('cuda', ...). Suppress only
# that compatibility warning so the original AMP execution path is unchanged.
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated.*",
)

from tqdm import tqdm
from torch.multiprocessing import Process, Queue
from hi2 import Hi2
from util.split_render_eval import eval_rendering_subset


def show_image(image, depth_prior, depth, normal):
    from util.utils import colorize_np
    image = image[[2,1,0]].permute(1, 2, 0).cpu().numpy()
    depth = colorize_np(np.concatenate((depth_prior.cpu().numpy(), depth.cpu().numpy()), axis=1), range=(0, 4))
    normal = normal.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('rgb / prior normal / aligned prior depth / JDSA depth', np.concatenate((image / 255.0, (normal[...,[2,1,0]]+1.)/2., depth), axis=1)[::2,::2])
    cv2.waitKey(1)


def mono_stream(queue, imagedir, calib, undistort=False, cropborder=False, start=0, length=100000):
    """ image generator """
    RES = 341 * 640

    calib = np.loadtxt(calib, delimiter=" ")
    K = np.array([[calib[0], 0, calib[2]],[0, calib[1], calib[3]],[0,0,1]])

    image_list = sorted(os.listdir(imagedir))[start:start+length]

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        intrinsics = torch.tensor(calib[:4])
        if len(calib) > 4 and undistort:
            image = cv2.undistort(image, K, calib[4:])
        if cropborder > 0:
            image = image[cropborder:-cropborder, cropborder:-cropborder]
            intrinsics[2:] -= cropborder

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((RES) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((RES) / (h0 * w0)))
        h1 = h1 - h1 % 8
        w1 = w1 - w1 % 8
        image = cv2.resize(image, (w1, h1))
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics[[0,2]] *= (w1 / w0)
        intrinsics[[1,3]] *= (h1 / h0)

        is_last = (t == len(image_list)-1)
        queue.put((t, image[None], intrinsics[None], is_last))

    time.sleep(10)


def trajectory_timestamps(imagedir, start=0):
    return np.array([
        float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1])
        for x in sorted(os.listdir(imagedir))[start:]
    ])[..., np.newaxis]


def save_full_trajectory(traj_full, imagedir, output, filename, start=0):
    tstamps_full = trajectory_timestamps(imagedir, start=start)
    ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)
    np.savetxt(os.path.join(output, filename), ttraj_full)


def save_trajectory(hi2, traj_full, imagedir, output, start=0):
    t = hi2.video.counter.value
    tstamps = hi2.video.tstamp[:t]
    poses_wc = lietorch.SE3(hi2.video.poses[:t]).inv().data
    np.save("{}/intrinsics.npy".format(output), hi2.video.intrinsics[0].cpu().numpy()*8)

    tstamps_full = trajectory_timestamps(imagedir, start=start)
    tstamps_kf = tstamps_full[tstamps.cpu().numpy().astype(int)]
    ttraj_kf = np.concatenate([tstamps_kf, poses_wc.cpu().numpy()], axis=1)
    np.savetxt(f"{output}/traj_kf.txt", ttraj_kf)                     # for evo evaluation
    if traj_full is not None:
        ttraj_full = np.concatenate([tstamps_full[:len(traj_full)], traj_full], axis=1)
        np.savetxt(f"{output}/traj_full.txt", ttraj_full)


def make_split(num_frames, every, offset):
    if every <= 0:
        return list(range(num_frames)), []
    if offset < 0 or offset >= every:
        raise ValueError("--holdout-offset must satisfy 0 <= offset < --holdout-every")
    test = [i for i in range(num_frames) if i % every == offset]
    test_set = set(test)
    train = [i for i in range(num_frames) if i not in test_set]
    if not train or train[0] != 0:
        raise ValueError("The split must keep frame 0 in the training/mapping set")
    return train, test


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--config", type=str, help="path to configuration file")
    parser.add_argument("--output", default='outputs/demo', help="path to save output")
    parser.add_argument("--gtdepthdir", type=str, default=None, help="optional for evaluation, assumes 16-bit depth scaled by 6553.5")

    parser.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "pretrained_models/droid.pth"))
    parser.add_argument("--buffer", type=int, default=-1, help="number of keyframes to buffer (default: 1/10 of total frames)")
    parser.add_argument("--undistort", action="store_true", help="undistort images if calib file contains distortion parameters")
    parser.add_argument("--cropborder", type=int, default=0, help="crop images to remove black border")

    parser.add_argument("--droidvis", action="store_true")
    parser.add_argument("--gsvis", action="store_true")

    parser.add_argument("--start", type=int, default=0, help="start frame")
    parser.add_argument("--length", type=int, default=100000, help="number of frames to process")

    # Evaluation-protocol options. Defaults preserve the original HI-SLAM2 path.
    parser.add_argument("--holdout-every", type=int, default=0, help="hold out one frame every N frames from mapping; 0 disables")
    parser.add_argument("--holdout-offset", type=int, default=4, help="held-out residue within each N-frame group")
    parser.add_argument("--dual-stage-eval", action="store_true", help="evaluate/save the online pre-global map and the official final map in one run")

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    torch.multiprocessing.set_start_method('spawn')

    all_files = sorted(os.listdir(args.imagedir))
    N = len(all_files)
    run_N = min(args.length, max(0, N - args.start))
    if run_N <= 0:
        raise RuntimeError("No input frames selected")

    train_indices, test_indices = make_split(run_N, args.holdout_every, args.holdout_offset)
    test_set = set(test_indices)
    last_train_idx = train_indices[-1]

    if args.holdout_every > 0:
        with open(os.path.join(args.output, "split.json"), "w", encoding="utf-8") as f:
            json.dump({
                "num_frames": run_N,
                "holdout_every": args.holdout_every,
                "holdout_offset": args.holdout_offset,
                "train_count": len(train_indices),
                "test_count": len(test_indices),
                "train_indices": train_indices,
                "test_indices": test_indices,
            }, f, indent=2)

    hi2 = None
    queue = Queue(maxsize=8)
    reader = Process(target=mono_stream, args=(queue, args.imagedir, args.calib, args.undistort, args.cropborder, args.start, args.length))
    reader.start()

    if args.buffer < 0:
        if args.holdout_every > 0 or args.dual_stage_eval:
            # Benchmark runs can retain substantially more mapping keyframes than
            # the original heuristic predicts, and terminate() can temporarily
            # insert supplemental non-keyframes. Reserve one slot per selected
            # input frame plus a small guard margin. This changes capacity only;
            # it does not change keyframe selection or any optimization logic.
            args.buffer = run_N + 32
        else:
            args.buffer = min(1000, N // 10 + 150)
    print(f"DepthVideo buffer: {args.buffer}")

    pbar = tqdm(total=run_N, desc="Processing keyframes")
    online_start = None

    while 1:
        (t, image, intrinsics, stream_is_last) = queue.get()
        pbar.update()

        if hi2 is None:
            args.image_size = [image.shape[2], image.shape[3]]
            hi2 = Hi2(args)
            torch.cuda.synchronize()
            online_start = time.perf_counter()

        allow_mapping = t not in test_set
        # For a split run, the final mapping frame (not necessarily the final
        # image) gets the same is_last treatment as an ordinary HI-SLAM2 run.
        mapping_is_last = (t == last_train_idx) if args.holdout_every > 0 else stream_is_last
        hi2.track(
            t,
            image,
            intrinsics=intrinsics,
            is_last=mapping_is_last,
            allow_mapping=allow_mapping,
        )

        if args.droidvis and hi2.video.counter.value > 0 and hi2.video.tstamp[hi2.video.counter.value-1] == t:
            from geom.ba import get_prior_depth_aligned
            index = hi2.video.counter.value-2
            depth_prior, _ = get_prior_depth_aligned(hi2.video.disps_prior_up[index][None].cuda(), hi2.video.dscales[index][None])
            show_image(image[0], 1./depth_prior.squeeze(), 1./hi2.video.disps_up[index], hi2.video.normals[index])
        pbar.set_description(f"Processing keyframe {hi2.video.counter.value} gs {hi2.gs.gaussians._xyz.shape[0]}")

        if stream_is_last:
            pbar.close()
            break

    torch.cuda.synchronize()
    online_seconds = time.perf_counter() - online_start
    reader.join()

    if args.dual_stage_eval:
        # ONLINE/PRE-GLOBAL SNAPSHOT -----------------------------------------
        # PoseTrajectoryFiller is the official HI-SLAM2 pose-only filler. It
        # temporarily estimates poses for non-keyframes/held-out frames but
        # does not optimize or change the Gaussian map.
        traj_online_internal = hi2.traj_filler(hi2.images)
        traj_online = traj_online_internal.inv().data.cpu().numpy()
        save_full_trajectory(traj_online, args.imagedir, args.output, "traj_online.txt", start=args.start)
        hi2.gs.gaussians.save_ply(os.path.join(args.output, "3dgs_online.ply"))
        online_gaussians = int(hi2.gs.gaussians._xyz.shape[0])

        online_train = eval_rendering_subset(
            hi2.images,
            traj_online_internal.matrix().data,
            train_indices,
            hi2.gs.gaussians,
            hi2.gs.background,
            hi2.gs.projection_matrix,
            hi2.gs.K,
            output_json=os.path.join(args.output, "metrics", "online_train.json"),
        )
        online_test = eval_rendering_subset(
            hi2.images,
            traj_online_internal.matrix().data,
            test_indices,
            hi2.gs.gaussians,
            hi2.gs.background,
            hi2.gs.projection_matrix,
            hi2.gs.K,
            output_json=os.path.join(args.output, "metrics", "online_test.json"),
        )

        # OFFICIAL FINAL PIPELINE -------------------------------------------
        # terminate() still performs the original supplemental-keyframe step,
        # global BA, Gaussian pose update, and global color refinement. Its
        # built-in mixed rendering eval is skipped only because the explicit
        # train/test evaluator below replaces that redundant evaluation.
        torch.cuda.synchronize()
        final_start = time.perf_counter()
        traj = hi2.terminate(run_builtin_eval=False)
        torch.cuda.synchronize()
        final_extra_seconds = time.perf_counter() - final_start
        save_trajectory(hi2, traj, args.imagedir, args.output, start=args.start)

        traj_final_c2w = torch.as_tensor(traj, dtype=torch.float32, device="cuda")
        traj_final_w2c = lietorch.SE3(traj_final_c2w).inv().matrix().data
        final_gaussians = int(hi2.gs.gaussians._xyz.shape[0])
        final_train = eval_rendering_subset(
            hi2.images,
            traj_final_w2c,
            train_indices,
            hi2.gs.gaussians,
            hi2.gs.background,
            hi2.gs.projection_matrix,
            hi2.gs.K,
            output_json=os.path.join(args.output, "metrics", "final_train.json"),
        )
        final_test = eval_rendering_subset(
            hi2.images,
            traj_final_w2c,
            test_indices,
            hi2.gs.gaussians,
            hi2.gs.background,
            hi2.gs.projection_matrix,
            hi2.gs.K,
            output_json=os.path.join(args.output, "metrics", "final_test.json"),
        )

        benchmark = {
            "num_frames": run_N,
            "train_count": len(train_indices),
            "test_count": len(test_indices),
            "online": {
                "train": online_train,
                "test": online_test,
                "gaussians": online_gaussians,
                "algorithm_seconds": online_seconds,
                "fps": run_N / online_seconds,
            },
            "final": {
                "train": final_train,
                "test": final_test,
                "gaussians": final_gaussians,
                "final_extra_seconds": final_extra_seconds,
                "algorithm_seconds": online_seconds + final_extra_seconds,
                "fps": run_N / (online_seconds + final_extra_seconds),
            },
        }
        with open(os.path.join(args.output, "benchmark_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(benchmark, f, indent=2)

        print("=" * 80)
        print("Dual-stage benchmark complete")
        print(f"Online: train {online_train['mean_psnr']:.4f}/{online_train['mean_ssim']:.6f}, "
              f"test {online_test['mean_psnr']:.4f}/{online_test['mean_ssim']:.6f}, "
              f"FPS {benchmark['online']['fps']:.4f}, Gaussians {online_gaussians}")
        print(f"Final : train {final_train['mean_psnr']:.4f}/{final_train['mean_ssim']:.6f}, "
              f"test {final_test['mean_psnr']:.4f}/{final_test['mean_ssim']:.6f}, "
              f"effective FPS {benchmark['final']['fps']:.4f}, Gaussians {final_gaussians}")
    else:
        traj = hi2.terminate()
        save_trajectory(hi2, traj, args.imagedir, args.output, start=args.start)

    print("Done")
