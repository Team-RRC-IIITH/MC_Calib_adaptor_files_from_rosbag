#!/usr/bin/env python3
"""Extract time-synchronized image sequences from a ROS 2 bag (mcap/db3) into
the Cam_001/, Cam_002/, ... folder layout expected by MC-Calib.

Requires a sourced ROS 2 install (for rosbag2_py, cv_bridge, rclpy) and must be
run with the *system* python3 that matches the ROS distro's Python ABI, e.g.:

    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 python_utils/rosbag_to_images.py \\
        --bag /home/soumoroy/data_collection/rosbag2_mccalib \\
        --out /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib \\
        --topics /camera1/cam_1/color/image_raw /camera2/cam_2/color/image_raw

Each --topics entry becomes Cam_001, Cam_002, ... in that order. Frames are
matched across cameras by nearest timestamp (within --slop seconds); a frame
is only saved if every camera has a match, so all output folders stay the
same length and in sync.
"""
import argparse
import bisect
from pathlib import Path

import cv2
import rclpy.serialization
import rosbag2_py
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


def read_images(bag_path: str, topic: str):
    """Return [(t_ns, cv_image), ...] for one topic, sorted by time."""
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    bridge = CvBridge()
    out = []
    while reader.has_next():
        _topic_name, data, t = reader.read_next()
        msg = rclpy.serialization.deserialize_message(data, Image)
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        out.append((t, cv_img))
    out.sort(key=lambda x: x[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="path to the bag directory (contains metadata.yaml)")
    ap.add_argument("--out", required=True, help="output root_path for MC-Calib")
    ap.add_argument("--topics", nargs="+", required=True,
                     help="one image topic per camera, in the order Cam_001, Cam_002, ...")
    ap.add_argument("--slop", type=float, default=0.02,
                     help="max time diff (s) to accept a cross-camera match (default 0.02)")
    ap.add_argument("--ext", default="png", help="output image extension (png/jpg/...)")
    args = ap.parse_args()

    slop_ns = int(args.slop * 1e9)
    out_root = Path(args.out)

    print("Reading messages ...")
    per_cam = [read_images(args.bag, t) for t in args.topics]
    for t, imgs in zip(args.topics, per_cam):
        print(f"  {t}: {len(imgs)} frames")
    if any(len(imgs) == 0 for imgs in per_cam):
        raise SystemExit("One or more topics had zero messages - check --topics against `ros2 bag info`.")

    cam_dirs = []
    for i in range(len(args.topics)):
        d = out_root / f"Cam_{i + 1:03d}"
        d.mkdir(parents=True, exist_ok=True)
        cam_dirs.append(d)

    saved = 0
    for t_ref, img_ref in per_cam[0]:
        matched = [(t_ref, img_ref)]
        ok = True
        for cam_idx in range(1, len(per_cam)):
            times = [t for t, _ in per_cam[cam_idx]]
            pos = bisect.bisect_left(times, t_ref)
            candidates = [p for p in (pos - 1, pos) if 0 <= p < len(times)]
            if not candidates:
                ok = False
                break
            best = min(candidates, key=lambda p: abs(times[p] - t_ref))
            if abs(times[best] - t_ref) > slop_ns:
                ok = False
                break
            matched.append(per_cam[cam_idx][best])
        if not ok:
            continue
        idx = saved + 1
        for cam_i, (_, img) in enumerate(matched):
            fname = cam_dirs[cam_i] / f"{idx:06d}.{args.ext}"
            cv2.imwrite(str(fname), img)
        saved += 1

    print(f"Saved {saved} synchronized frames per camera into {out_root}")
    for d in cam_dirs:
        print(f"  {d} ({saved} images)")


if __name__ == "__main__":
    main()
