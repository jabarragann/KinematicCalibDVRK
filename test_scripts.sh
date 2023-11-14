#!/bin/bash

set -x


# ----------------------------------------------------------------
# Experiment: Calculate error with different handeye calib files
#-----------------------------------------------------------------

# python scripts/data_collection_scripts/calculate_robot_real_pose.py  \
# --data_file data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-33-49/combined_data.csv \
# --handeye_file ./data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-33-49/hand_eye_calib.json

python scripts/data_collection_scripts/calculate_robot_real_pose.py \
--data_file data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-33-49/combined_data.csv \
--handeye_file ./data/experiments/data_collection2_updated_collection_scripts/13-11-2023-20-53-03/hand_eye_calib.json

