# MC_Calib_adaptor_files_from_rosbag
putting adaptor scripts for calibration

## Running MC-Calib
- It finished in ~2 seconds with mean reprojection error: 0.408
```
cd /home/soumoroy/MC-Calib/build
./apps/calibrate/calibrate ../configs/Real_images/calib_param_intel_realsens455.yml
```

## rosbag2 to images MC-Calib
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
