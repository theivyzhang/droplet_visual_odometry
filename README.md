# Visual Odometry Package for Underwater Robot Navigation
Copyright @ 2023 Dartmouth Robotics Lab

## 1. Running Unit Testing 
Run the following command to save consecutive rosbag frames and translation & quaternion data
1. Play your rosbag in the background
2. Ensure that "/camera_array/cam0/image_raw/compressed" and "/bluerov_controller/ar_tag_detector" are available topics (run `rostopic list` to check)
3. GROUND TRUTH: Run `python trajectory_evaluation_ground_truth.py` which produces a txt file called "stamped_ground_truth.txt"
4. VISUAL ODOMETRY: Run `python trajectory_evaluation_vis_odom_extraction.py` which produces a txt file called "stamped_traj_estimates.txt"

## 2. Running Trajectory Evaluation
