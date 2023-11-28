# Recordings meta data 

There are two types of csv files: `raw_sensor` and `actual_state` files. The former are generated when collecting data from sensors.
The former ones are generated after calculating the robot actual state.

Naming convention: `<File-type>-<Traj-type>-<Order>-<RosBag>.csv`
For example: `raw_sensor_soft_01.csv`, `actual_state_rosbag_03_traj1.csv`

# Processing steps


1. Join all the files of each recording into a single one. Manually apply naming convention.

This can be achieved for each subfolder in collection with 
```bash
find ./data/experiments/data_collection4_or
ig -mindepth 1 -type d -exec python3 ./scripts/util_scripts/combine_data_files.py --data_dir {} \;
```

2. Combine all soft data into a joint file and calculate the hand-eye calibration
```bash
python scripts/data_collection_scripts/perform_hand_eye_calib.py --data_path  ./data/experiments/data_collection4/combined_softs/combined_data.csv
```

3. Calculate actual states from handeye calibration from the combined file created in step 2.
File by file
```bash
python scripts/data_collection_scripts/calculate_robot_real_pose.py --data_file ./data/experiments/data_collection4/raw_sensor_soft_01.csv --handeye_file ./data/experiments/data_collection4/combined_softs/hand_eye_calib.json
```

Batch processing
```bash
find data/experiments/data_collection4/ -maxdepth 1 -type f -name '*raw_sensor*' -exec python scripts/data_collection_scripts/calculate_robot_real_pose.py --data_file {} --handeye_file ./data/experiments/data_collection4/combined_softs/hand_eye_calib.json \;
```

4. Train models with actual states files. 
Select a dataset from a config group with +GROUP=OPTION
```bash
python scripts/learning_scripts/train_test_simple_net.py +path_config=juanubuntu_dataset4
```

5. Test models with either `raw_sensor` files or `actual_state` files 

6. Sanity check: compare that hand-eye calibration calculated with only data from a rosbag trajectory will lead to the same values. Make sure that marker was not moved during collection

```bash
python scripts/util_scripts/calc_hand_eye_differences.py --path1 data/experiments/data_collection4_orig/15-11-2023-19-19-00 --path2 data/experiments/data_collection4_orig/15-11-2023-18-08-28
```

## Recordings meta data

### Collection 3

* Normal tool 
* Total soft: 3200

| Order   | trajectory type   | Rosbag   | size   | timestamp         |
| ------- | ----------------- | -------- | ------ | ----------------- |
| 1       | Soft              |          | 802    | 14112023-183732   |
| 2       | Rosbag            | traj1    | 342    | 14112023-184931   |
| 3       | Soft              |          | 802    | 14112023-185746   |
| 4       | Rosbag            | traj1    | 342    | 14112023-190851   |
| 5       | Soft              |          | 802    | 14112023-191355   |
| 6       | Rosbag            | traj1    | 342    | 14112023-192500   |
| 7       | Soft              |          | 802    | 14112023-193049   |
| ------- | ----------------- | -------- | ------ | ----------------- |

### Collection 4

* Normal tool
* Total soft: 4000 

| Order   | trajectory type   | Rosbag   | size   | timestamp         |
| ------- | ----------------- | -------- | ------ | ----------------- |
| 1       | Soft              |          | 400    | 15112023-175655   |
| 2       | Soft              |          | 400    | 15112023-180300   |
| 3       | Rosbag            | traj1    | 324    | 15112023-180828   |
| 4       | Soft              |          | 800    | 15112023-181222   |
| 5       | Soft              |          | 800    | 15112023-182402   |
| 6       | Rosbag            | traj1    | 342    | 15112023-183448   |
| 7       | Soft              |          | 800    | 15112023-184152   |
| 8       | Rosbag            | traj1    | 342    | 15112023-185302   |
| 9       | Rosbag            | traj3    | 520    | 15112023-185933   |
| 10      | Rosbag            | traj1    | 342    | 15112023-190416   |
| 11      | Soft              |          | 800    | 15112023-190827   |
| 12      | Rosbag            | traj1    | 342    | 15112023-191900   |
| ------- | ----------------- | -------- | ------ | ----------------- |

### Collection 5

* Lubricated tool.
* Total soft: 3600

| Order   | trajectory type   | Rosbag   | size   | timestamp         |
| ------- | ----------------- | -------- | ------ | ----------------- |
| 1       | Rosbag            | traj1    | 342    | 16112023-163416   |
| 2       | Rosbag            | traj3    | 520    | 16112023-163830   |
| 3       | Soft              |          | 2400   | 16112023-164340   |
| 4       | Soft              |          | 802    | 16112023-171452   |
| 5       | Rosbag            | traj1    | 342    | 16112023-172637   |
| 6       | Soft              |          | 400    | 16112023-173106   |
| ------- | ----------------- | -------- | ------ | ----------------- |




