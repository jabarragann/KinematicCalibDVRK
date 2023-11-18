#!/bin/bash

set -x


# ----------------------------------------------------------------
# Experiment: Calculate error with different handeye calib files
#-----------------------------------------------------------------

# python scripts/data_collection_scripts/calculate_robot_real_pose.py  \
# --data_file data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-33-49/combined_data.csv \
# --handeye_file ./data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-33-49/hand_eye_calib.json

python scripts/data_collection_scripts/calculate_robot_real_pose.py  \
--data_file data/experiments/data_collection3_orig/14-11-2023-18-37-32/record_001.csv \
--handeye_file ./data/experiments/data_collection3/combined/hand_eye_calib.json


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
