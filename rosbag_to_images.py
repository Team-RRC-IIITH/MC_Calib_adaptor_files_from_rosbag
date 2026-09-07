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

Also writes <out>/realsense_factory_intrinsics.yml (camera_matrix +
distortion_vector per camera, read from each topic's sibling /camera_info),
in the format MC-Calib's cam_params_path expects - point your config's
cam_params_path directly at this file. Pass --no-intrinsics to skip it (e.g.
if a camera_info topic isn't actually available).
"""
import argparse
import bisect
from pathlib import Path

import cv2
import numpy as np
import rclpy.serialization
import rosbag2_py
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


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


def read_first_camera_info(bag_path: str, topic: str) -> CameraInfo:
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    if not reader.has_next():
        raise RuntimeError(f"no messages found on {topic}")
    _topic_name, data, _t = reader.read_next()
    return rclpy.serialization.deserialize_message(data, CameraInfo)


def camera_info_topic(image_topic: str) -> str:
    """.../color/image_raw -> .../color/camera_info (same namespace, sibling topic)."""
    return image_topic.rsplit("/", 1)[0] + "/camera_info"


def write_intrinsics_yml(path: Path, bag_path: str, image_topics):
    """Write one camera_<i> entry per topic (camera_matrix + distortion_vector),
    read from each topic's sibling camera_info - the format MC-Calib's
    cam_params_path expects."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    for i, image_topic in enumerate(image_topics):
        info_topic = camera_info_topic(image_topic)
        ci = read_first_camera_info(bag_path, info_topic)
        K = np.array(ci.k, dtype=np.float64).reshape(3, 3)
        D = np.array(ci.d, dtype=np.float64).reshape(1, -1)
        fs.startWriteStruct(f"camera_{i}", cv2.FileNode_MAP)
        fs.write("camera_matrix", K)
        fs.write("distortion_vector", D)
        fs.endWriteStruct()
        print(f"  camera_{i}: intrinsics from {info_topic} ({ci.distortion_model}, {len(ci.d)} coeffs)")
    fs.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="path to the bag directory (contains metadata.yaml)")
    ap.add_argument("--out", required=True, help="output root_path for MC-Calib")
    ap.add_argument("--topics", nargs="+", required=True,
                     help="one image topic per camera, in the order Cam_001, Cam_002, ...")
    ap.add_argument("--slop", type=float, default=0.02,
                     help="max time diff (s) to accept a cross-camera match (default 0.02)")
    ap.add_argument("--ext", default="png", help="output image extension (png/jpg/...)")
    ap.add_argument("--no-intrinsics", action="store_true",
                     help="skip writing realsense_factory_intrinsics.yml")
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

    if not args.no_intrinsics:
        intrinsics_path = out_root / "realsense_factory_intrinsics.yml"
        print(f"Writing intrinsics to {intrinsics_path} ...")
        try:
            write_intrinsics_yml(intrinsics_path, args.bag, args.topics)
            print(f"Wrote {intrinsics_path} - point cam_params_path at this file.")
        except RuntimeError as e:
            print(f"Skipped intrinsics ({e}). Pass --no-intrinsics to silence this, "
                  f"or check the topic has a sibling /camera_info.")


if __name__ == "__main__":
    main()
