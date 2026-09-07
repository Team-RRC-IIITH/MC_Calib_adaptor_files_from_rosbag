# MC_Calib_adaptor_files_from_rosbag
putting adaptor scripts for calibration

## Running MC-Calib
- It finished in ~2 seconds with mean reprojection error: 0.19
```
cd /home/soumoroy/MC-Calib/build
./apps/calibrate/calibrate ../configs/Real_images/calib_param_intel_realsens455.yml
```

## rosbag2 to images MC-Calib format
```
source /opt/ros/jazzy/setup.zsh                                                                                                         
python3 /home/soumoroy/MC-Calib/python_utils/rosbag_to_images.py \
    --bag /home/soumoroy/data_collection/rosbag2_mccalib \
    --out /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib \
    --topics /camera1/cam_1/color/image_raw /camera2/cam_2/color/image_raw
```

## reconstruct pointcloud
```
source /opt/ros/jazzy/setup.zsh
python3 python_utils/reconstruct_pointcloud.py \
    --bag /home/soumoroy/data_collection/rosbag2_mccalib \
    --calib /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib/realsense_2cam_calibrated.yml \
    --out /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib/fused_cloud.ply \
    --preview /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib/fused_cloud_preview.png
```

## Per Cam PCD exporter
```
/usr/bin/python3 export_camera_clouds.py \
    --bag /home/soumoroy/data_collection/rosbag2_mccalib_objects \ 
    --calib /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib_static_v2/realsense_cam2_objects_calibrated.yml \ 
    --out-prefix /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/cam \
    --max-depth 1.2
```

## Coloured ICP
```
cd /home/soumoroy/MC-Calib/python_utils
python3 colored_icp_refine.py \
    --cloud0 /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/cam_0.ply \
    --cloud1 /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/cam_1.ply \
    --calib /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib_static_v2/realsense_cam2_objects_calibrated.yml \
    --out /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/refined_pose.yml \
    --preview /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/icp_preview.png \
    --combined-out /home/soumoroy/MC-Calib/data/Real_Images/rosbag2_mccalib_objects/icp/combined_icp_cloud.ply
```
