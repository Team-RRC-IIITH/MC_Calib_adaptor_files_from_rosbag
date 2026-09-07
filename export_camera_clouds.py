#!/usr/bin/env python3
"""Export each camera's RGB-D point cloud in its OWN local (color-optical)
frame — i.e. without applying MC-Calib's camera_pose_matrix — so a separate
stage (colored_icp_refine.py, which needs open3d) can register them and
cross-check/refine the extrinsic independently of MC-Calib's own bundle
adjustment.

This is the ROS-dependent half of the pipeline; run it with a sourced ROS 2
install, same as reconstruct_pointcloud.py:

    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 python_utils/export_camera_clouds.py \\
        --bag /home/soumoroy/data_collection/rosbag2_mccalib \\
        --calib /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib/realsense_2cam_calibrated.yml \\
        --out-prefix /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib/icp/cam \\
        --max-depth 1.2

Writes <out-prefix>_0.ply, <out-prefix>_1.ply, ... (one per --cam-prefixes
entry, in each camera's own frame). Pass the same --calib yml to
colored_icp_refine.py afterwards - it reads the init transform from it.
"""
import argparse
from pathlib import Path

from reconstruct_pointcloud import load_calibration, process_camera, read_stamps, write_ply


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--calib", required=True, help="MC-Calib output yml (e.g. realsense_2cam_calibrated.yml)")
    ap.add_argument("--out-prefix", required=True, help="output files are <out-prefix>_<cam_idx>.ply")
    ap.add_argument(
        "--cam-prefixes",
        nargs="+",
        default=["camera1/cam_1", "camera2/cam_2"],
        help="ROS namespace prefix per camera, in the SAME order as camera_0, camera_1, ... in --calib",
    )
    ap.add_argument(
        "--time-frac",
        type=float,
        default=0.5,
        help="fraction [0,1] into camera_prefixes[0]'s depth stream to sample the frame from (default: middle)",
    )
    ap.add_argument(
        "--max-depth",
        type=float,
        default=1.2,
        help="clip points farther than this (m). Keep this tight (just past the shared desk/board region) - "
        "far, noisy depth hurts ICP convergence more than it helps.",
    )
    ap.add_argument("--stride", type=int, default=1, help="pixel stride for backprojection (>1 to subsample/speed up)")
    args = ap.parse_args()

    cams = load_calibration(args.calib, len(args.cam_prefixes))
    ref_stamps = read_stamps(args.bag, f"/{args.cam_prefixes[0]}/depth/image_rect_raw")
    target_ns = ref_stamps[int(args.time_frac * (len(ref_stamps) - 1))]

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for i, prefix in enumerate(args.cam_prefixes):
        pts, rgb = process_camera(
            args.bag, prefix, cams[i], target_ns, args.max_depth, args.stride, apply_pose=False
        )
        out_path = f"{out_prefix}_{i}.ply"
        write_ply(out_path, pts, rgb)
        print(f"camera_{i} ({prefix}): {pts.shape[0]} points -> {out_path}")


if __name__ == "__main__":
    main()
