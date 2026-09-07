#!/usr/bin/env python3
"""Cross-check (and optionally refine) MC-Calib's extrinsic between two
cameras by registering their RGB-D point clouds with multi-scale colored
ICP (Park et al., 2017 - the same algorithm behind
open3d.pipelines.registration.registration_colored_icp).

IMPORTANT: this optimizes 3D depth+color alignment, a DIFFERENT objective
than MC-Calib's 2D Charuco-corner reprojection error. A low ICP fitness/RMSE
here means the depth data is geometrically consistent with MC-Calib's
extrinsic (good for depth-fusion/reconstruction use); it is not the same
number as - and should not be reported as - MC-Calib's own reprojection
error, and there is no guarantee the ICP-refined pose would score better on
that 2D metric (depth sensors are generally noisier than sub-pixel corner
detections for this kind of rigid-pose estimation).

Run this AFTER export_camera_clouds.py, with a plain (non-ROS) python3 that
has open3d + opencv-python installed, e.g. your anaconda env:

    python3 colored_icp_refine.py \
    --cloud0 /home/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/cam_0.ply \
    --cloud1 /home/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/cam_1.ply \
    --calib /home/MC-Calib/data/Real_Images/rosbag2_mccalib_static_v2/realsense_cam2_objects_calibrated.yml \
    --out /home/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/refined_pose.yml \
    --preview /home/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/icp_preview.png \
    --combined-out /home/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/combined_icp_cloud.ply
"""
import argparse

import cv2
import numpy as np
import open3d as o3d


def load_init_transform(calib_path, cam_index=1):
    fs = cv2.FileStorage(str(calib_path), cv2.FILE_STORAGE_READ)
    node = fs.getNode(f"camera_{cam_index}")
    if node.empty():
        raise RuntimeError(f"'camera_{cam_index}' not found in {calib_path}")
    T = node.getNode("camera_pose_matrix").mat()
    fs.release()
    return T


def pose_diff(T_a, T_b):
    """Rotation-angle (deg) and translation (mm) difference between two 4x4 poses."""
    R_rel = T_a[:3, :3].T @ T_b[:3, :3]
    trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(trace))
    trans_mm = np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]) * 1000.0
    return angle_deg, trans_mm


def multiscale_colored_icp(source, target, init, voxel_radii=(0.04, 0.02, 0.01), max_iters=(50, 30, 14)):
    """Standard multi-scale colored-ICP loop (Open3D's own recipe): coarse-to-fine
    voxel downsampling, re-estimating normals at each scale, chaining the
    transformation from the previous scale as the next scale's init."""
    current = init.copy()
    result = None
    for scale, (radius, n_iter) in enumerate(zip(voxel_radii, max_iters)):
        src_down = source.voxel_down_sample(radius)
        tgt_down = target.voxel_down_sample(radius)
        src_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))
        tgt_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2, max_nn=30))

        result = o3d.pipelines.registration.registration_colored_icp(
            src_down,
            tgt_down,
            radius,
            current,
            o3d.pipelines.registration.TransformationEstimationForColoredICP(),
            o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=n_iter),
        )
        current = result.transformation
        print(
            f"  scale {scale} (voxel={radius:.3f} m, {n_iter} iters): "
            f"fitness={result.fitness:.4f}  inlier_rmse={result.inlier_rmse * 1000:.2f} mm  "
            f"points: src={len(src_down.points)} tgt={len(tgt_down.points)}"
        )
    return current, result


def save_preview(path, source, target, T_init, T_refined):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tgt_pts = np.asarray(target.points)
    tgt_col = np.asarray(target.colors)
    src_pts = np.asarray(source.points)
    src_col = np.asarray(source.colors)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, T, title in zip(axes, [T_init, T_refined], ["MC-Calib extrinsic (init)", "Colored-ICP refined"]):
        pts_aligned = (T[:3, :3] @ src_pts.T).T + T[:3, 3]
        ax.scatter(tgt_pts[:, 0], tgt_pts[:, 2], s=0.3, c=tgt_col, label="camera_0 (target)")
        ax.scatter(pts_aligned[:, 0], pts_aligned[:, 2], s=0.3, c=src_col, label="camera_1 (source, aligned)")
        ax.set_title(title)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m, depth)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.invert_yaxis()
        ax.legend(markerscale=20)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cloud0", required=True, help="camera_0's cloud in its own frame (target/reference)")
    ap.add_argument("--cloud1", required=True, help="camera_1's cloud in its own frame (source, gets registered)")
    ap.add_argument("--calib", required=True, help="MC-Calib output yml, for the init transform (camera_1's camera_pose_matrix)")
    ap.add_argument("--out", default=None, help="optional: write the refined 4x4 transform + pose diff to this yml")
    ap.add_argument("--preview", default=None, help="optional PNG comparing init vs. ICP-refined alignment")
    ap.add_argument(
        "--combined-out",
        default=None,
        help="optional: write the fused point cloud (target + ICP-aligned source) to this .ply",
    )
    args = ap.parse_args()

    target = o3d.io.read_point_cloud(args.cloud0)
    source = o3d.io.read_point_cloud(args.cloud1)
    print(f"camera_0 (target): {len(target.points)} points")
    print(f"camera_1 (source): {len(source.points)} points")

    T_init = load_init_transform(args.calib, cam_index=1)
    print("\nMC-Calib init transform (camera_1 -> camera_0):")
    print(T_init)

    print("\nRunning multi-scale colored ICP ...")
    T_refined, result = multiscale_colored_icp(source, target, T_init)

    angle_deg, trans_mm = pose_diff(T_init, T_refined)
    print("\nRefined transform (camera_1 -> camera_0):")
    print(T_refined)
    print(f"\nFinal fitness (overlap fraction, higher=better): {result.fitness:.4f}")
    print(f"Final inlier RMSE (3D, mm): {result.inlier_rmse * 1000:.2f}")
    print(f"Delta vs. MC-Calib extrinsic: {angle_deg:.3f} deg rotation, {trans_mm:.2f} mm translation")
    print(
        "\nNote: fitness/inlier_rmse are 3D depth-cloud metrics, NOT comparable to "
        "MC-Calib's 2D reprojection error (see module docstring)."
    )

    if args.out:
        fs = cv2.FileStorage(args.out, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_pose_matrix_init", T_init)
        fs.write("camera_pose_matrix_icp_refined", T_refined)
        fs.write("icp_fitness", result.fitness)
        fs.write("icp_inlier_rmse_m", result.inlier_rmse)
        fs.write("delta_rotation_deg", angle_deg)
        fs.write("delta_translation_mm", trans_mm)
        fs.release()
        print(f"\nWrote {args.out}")

    if args.preview:
        save_preview(args.preview, source, target, T_init, T_refined)
        print(f"Wrote {args.preview}")

    if args.combined_out:
        combined = o3d.geometry.PointCloud(target)
        combined += o3d.geometry.PointCloud(source).transform(T_refined)
        o3d.io.write_point_cloud(args.combined_out, combined)
        print(f"Wrote {len(combined.points)}-point combined cloud to {args.combined_out}")


if __name__ == "__main__":
    main()
