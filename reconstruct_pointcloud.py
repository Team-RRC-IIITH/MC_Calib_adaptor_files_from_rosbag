#!/usr/bin/env python3
"""Fuse the RGB-D streams from both RealSense cameras into a single point
cloud, using MC-Calib's extrinsic/intrinsic calibration result, so the
calibration can be checked visually: if the extrinsics are good, the two
cameras' point clouds should overlap into one consistent surface (desk,
board, ...) rather than showing a doubled/offset scene.

Requires a sourced ROS 2 install (rosbag2_py, cv_bridge, rclpy), run with the
system python3 matching the ROS distro's ABI:

    source /opt/ros/jazzy/setup.bash
    /usr/bin/python3 python_utils/reconstruct_pointcloud.py \\
        --bag /home/soumoroy/data_collection/rosbag2_mccalib \\
        --calib /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib/realsense_2cam_calibrated.yml \\
        --out /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib/fused_cloud.ply \\
        --preview /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib/fused_cloud_preview.png

Convention: MC-Calib's `camera_pose_matrix` for camera_i is the camera-i ->
world transform (world == camera_0, whose pose_matrix is Identity), i.e.
X_world = R_i @ X_cam_i + t_i. This matches how python_utils/display_calib_
cameras.py uses the same field, so camera_1's points only need this single
matrix applied directly (no inversion) to land in camera_0's frame.
"""
import argparse

import cv2
import numpy as np
import rclpy.serialization
import rosbag2_py
from cv_bridge import CvBridge
from realsense2_camera_msgs.msg import Extrinsics
from sensor_msgs.msg import CameraInfo, Image


def _open_reader(bag_path, topic):
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))
    return reader


def read_stamps(bag_path, topic):
    """Return the list of message timestamps (ns) on `topic`, in bag order."""
    reader = _open_reader(bag_path, topic)
    stamps = []
    while reader.has_next():
        _topic, _data, stamp = reader.read_next()
        stamps.append(stamp)
    return stamps


def read_nearest(bag_path, topic, msg_type, target_ns=None):
    """Return the deserialized message on `topic` nearest to target_ns (or the
    first message if target_ns is None)."""
    reader = _open_reader(bag_path, topic)
    best_data, best_dt = None, None
    while reader.has_next():
        _topic, data, stamp = reader.read_next()
        if target_ns is None:
            return rclpy.serialization.deserialize_message(data, msg_type)
        dt = abs(stamp - target_ns)
        if best_dt is None or dt < best_dt:
            best_dt, best_data = dt, data
    if best_data is None:
        raise RuntimeError(f"no messages found on topic {topic}")
    return rclpy.serialization.deserialize_message(best_data, msg_type)


def load_calibration(calib_path, num_cameras):
    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    cams = []
    for i in range(num_cameras):
        node = fs.getNode(f"camera_{i}")
        if node.empty():
            raise RuntimeError(f"'camera_{i}' not found in {calib_path}")
        cams.append(
            dict(
                K=node.getNode("camera_matrix").mat(),
                D=node.getNode("distortion_vector").mat(),
                T=node.getNode("camera_pose_matrix").mat(),
            )
        )
    fs.release()
    return cams


def backproject_depth(depth_m, K_depth):
    """depth_m: HxW depth in meters. Returns (HxWx3 points in the depth optical
    frame, HxW validity mask)."""
    h, w = depth_m.shape
    us, vs = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    x = (us - K_depth[0, 2]) * z / K_depth[0, 0]
    y = (vs - K_depth[1, 2]) * z / K_depth[1, 1]
    pts = np.stack([x, y, z], axis=-1)
    return pts, z > 0


def process_camera(bag_path, prefix, cam_calib, target_ns, max_depth, stride):
    depth_msg = read_nearest(bag_path, f"/{prefix}/depth/image_rect_raw", Image, target_ns)
    color_msg = read_nearest(bag_path, f"/{prefix}/color/image_raw", Image, target_ns)
    depth_info = read_nearest(bag_path, f"/{prefix}/depth/camera_info", CameraInfo)
    ext = read_nearest(bag_path, f"/{prefix}/extrinsics/depth_to_color", Extrinsics)

    bridge = CvBridge()
    depth = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough").astype(np.float32)
    depth /= 1000.0  # RealSense depth is uint16 millimeters -> meters
    color = bridge.imgmsg_to_cv2(color_msg, desired_encoding="bgr8")

    depth = depth[::stride, ::stride]
    K_depth = np.array(depth_info.k, dtype=np.float64).reshape(3, 3)
    K_depth[0, 0] /= stride
    K_depth[1, 1] /= stride
    K_depth[0, 2] /= stride
    K_depth[1, 2] /= stride

    # realsense2_camera_msgs/Extrinsics.rotation is a COLUMN-MAJOR 3x3.
    R_d2c = np.array(ext.rotation, dtype=np.float64).reshape(3, 3, order="F")
    t_d2c = np.array(ext.translation, dtype=np.float64)

    pts_depth, valid = backproject_depth(depth, K_depth)
    valid &= pts_depth[..., 2] <= max_depth
    pts_depth = pts_depth[valid]
    pts_color_frame = (R_d2c @ pts_depth.T).T + t_d2c

    K_color = cam_calib["K"]
    D_color = cam_calib["D"]
    # Undistort once with newCameraMatrix == K_color, so a plain pinhole
    # projection with K_color samples the correct pixel afterwards.
    color_undist = cv2.undistort(color, K_color, D_color)

    Z = pts_color_frame[:, 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        u = pts_color_frame[:, 0] * K_color[0, 0] / Z + K_color[0, 2]
        v = pts_color_frame[:, 1] * K_color[1, 1] / Z + K_color[1, 2]
    h, w = color_undist.shape[:2]
    in_view = (Z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)

    pts_color_frame = pts_color_frame[in_view]
    u = u[in_view].astype(np.int32)
    v = v[in_view].astype(np.int32)
    rgb = color_undist[v, u][:, ::-1]  # BGR -> RGB

    T = cam_calib["T"]
    pts_world = (T[:3, :3] @ pts_color_frame.T).T + T[:3, 3]
    return pts_world, rgb


def write_ply(path, pts, colors):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        np.savetxt(
            f,
            np.column_stack([pts, colors]),
            fmt="%.4f %.4f %.4f %d %d %d",
        )


def save_preview(path, all_pts, all_colors, max_points_per_cam=40000, rng=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = rng or np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, (pts, colors) in enumerate(zip(all_pts, all_colors)):
        n = pts.shape[0]
        if n > max_points_per_cam:
            idx = rng.choice(n, max_points_per_cam, replace=False)
            pts, colors = pts[idx], colors[idx]
        c = colors.astype(np.float32) / 255.0
        axes[0].scatter(pts[:, 0], pts[:, 2], s=0.3, c=c, label=f"camera_{i}")
        axes[1].scatter(pts[:, 0], pts[:, 1], s=0.3, c=c, label=f"camera_{i}")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Z (m, depth)")
    axes[0].set_title("Top-down (X-Z)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].set_title("Front (X-Y)")
    for ax in axes:
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.legend(markerscale=20)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag", required=True, help="path to the bag directory (contains metadata.yaml)")
    ap.add_argument("--calib", required=True, help="MC-Calib output yml (e.g. realsense_2cam_calibrated.yml)")
    ap.add_argument("--out", required=True, help="output .ply path")
    ap.add_argument("--preview", default=None, help="optional PNG with top-down/front scatter views")
    ap.add_argument(
        "--cam-prefixes",
        nargs="+",
        default=["camera1/cam_1", "camera2/cam_2"],
        help="ROS namespace prefix per camera, in the SAME order as camera_0, camera_1, ... in --calib "
        "(i.e. the same order as Cam_001, Cam_002, ... during extraction/calibration)",
    )
    ap.add_argument(
        "--time-frac",
        type=float,
        default=0.5,
        help="fraction [0,1] into camera_prefixes[0]'s depth stream to sample the frame from (default: middle)",
    )
    ap.add_argument("--max-depth", type=float, default=2.0, help="clip points farther than this (m)")
    ap.add_argument("--stride", type=int, default=1, help="pixel stride for backprojection (>1 to subsample/speed up)")
    args = ap.parse_args()

    cams = load_calibration(args.calib, len(args.cam_prefixes))

    ref_stamps = read_stamps(args.bag, f"/{args.cam_prefixes[0]}/depth/image_rect_raw")
    target_ns = ref_stamps[int(args.time_frac * (len(ref_stamps) - 1))]

    all_pts, all_colors = [], []
    for i, prefix in enumerate(args.cam_prefixes):
        pts, rgb = process_camera(args.bag, prefix, cams[i], target_ns, args.max_depth, args.stride)
        print(f"camera_{i} ({prefix}): {pts.shape[0]} points")
        all_pts.append(pts)
        all_colors.append(rgb)

    pts = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_colors, axis=0)
    write_ply(args.out, pts, colors)
    print(f"Wrote {pts.shape[0]} points to {args.out}")

    if args.preview:
        save_preview(args.preview, all_pts, all_colors)
        print(f"Wrote preview to {args.preview}")


if __name__ == "__main__":
    main()
