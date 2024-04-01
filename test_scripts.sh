#!/bin/bash

set -x


# ----------------------------------------------------------------
# Experiment: Calculate error with different handeye calib files
#-----------------------------------------------------------------

# python scripts/data_collection_scripts/calculate_robot_real_pose.py  \
# --data_file data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-33-49/combined_data.csv \
# --handeye_file ./data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-33-49/hand_eye_calib.json

# python scripts/data_collection_scripts/calculate_robot_real_pose.py  \
# --data_file data/experiments/data_collection3_orig/14-11-2023-18-37-32/record_001.csv \
# --handeye_file ./data/experiments/data_collection3/combined/hand_eye_calib.json


# python scripts/data_collection_scripts/calculate_robot_real_pose.py \
# --data_file data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-33-49/combined_data.csv \
# --handeye_file ./data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-53-03/hand_eye_calib.json

#---------------------------------------------------------------
# Generate training dataset
#---------------------------------------------------------------

# python scripts/data_collection_scripts/calculate_robot_real_pose.py \
# --handeye_file ./data/experiments/data_collection3/combined/hand_eye_calib.json \
# --data_file ./data/experiments/data_collection3/combined/record_001_2.csv \
# --out_name dataset2.csv
 

#---------------------------------------------------------------
# Generate dataset from collected data 03
#---------------------------------------------------------------

python scripts/data_collection_scripts/calculate_robot_real_pose.py  \
--data_file ./data/experiments/data_collection3/soft_traj4/record_001.csv \
--handeye_file ./data/experiments/data_collection3/combined/hand_eye_calib.json \
--out_name  dataset4.csv

#---------------------------------------------------------------
# Test network on trajectory 
#---------------------------------------------------------------

## Good results -> tres traj: raw_sensor_rosbag_08_traj1.csv, raw_sensor_rosbag_09_traj3.csv
## measured-setpoint model
python scripts/learning_scripts/test_network_on_trajectory.py \
--model_path outputs_hydra/train_test_simple_net_20231129_214006 \
--test_data_name raw_sensor_rosbag_09_traj3.csv 

## actual-measured model
python scripts/learning_scripts/test_network_on_trajectory.py \
--model_path outputs_hydra/train_test_simple_net_20231129_202216 \
--test_data_name raw_sensor_rosbag_09_traj3.csv 


#---------------------------------------------------------------
# Best models - plots for paper
#---------------------------------------------------------------

# measured-setpoint
python scripts/learning_scripts/test_network_on_trajectory.py \
--test_data_name raw_sensor_rosbag_09_traj3.csv \
--model_path outputs_hydra/train_test_simple_net_20231201_123942 \
--output_path results/paper_plots1

# actual-measured
python scripts/learning_scripts/test_network_on_trajectory.py \
--test_data_name raw_sensor_rosbag_09_traj3.csv \
--model_path outputs_hydra/train_test_simple_net_20231201_124659 \
--output_path results/paper_plots1