# Improving the realism of robotic surgery simulation through injection of learning-based estimated errors

## Setup environment - Last modified on August 03, 2024


* This code was tested with an anaconda virtual environment.
* Requires surgical robotics challenge version: 2.0.0

Installation of error injection code base.
```bash
conda create -n ki_err python=3.10 numpy
# The following command might install other pytorch version in the future. For now it is pytorch 2.4.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 
pip install -r requirements.txt #python requirements
pip install -r requirements_torch.txt # torch requirements
pip install -r requirements_ros.txt # install ros dependencies - PYKDL needs to be compiled usually so this will take some time ~ 2-5min
pip install -e . # install our package
```

Note: if installation of the `ros_requirements` fails, you can also install all the software in your main python interpreter that contains your native ROS installation.


## Deep learning model scripts 

### Setup
Download data from surgical robotics challenge [teams channel](https://livejohnshopkins.sharepoint.com/:f:/r/sites/Surgicalroboticschallenge/Shared%20Documents/dvrk%20error%20injection%20project/processed_data?csf=1&web=1&e=CF4JPc). This can be added anywhere in the project root folder.

Next, create a new path configuration file in `scripts/learning_scripts/train_test_simple_net_confs/path_config/`. This path sets the path from which data will be loaded. In the new file set the following two fields

```
workspace: "/home/juan95/research/accelnet_grant/KinematicCalibDVRK"
datapath: "${.workspace}/data/processed_data/data_collection4"
```

Change the main config file in `scripts/learning_scripts/train_test_simple_net_confs/train_test_simple_net.yaml` to point to the new path configuration file. You only need to change the `defaults` field to point to the new path configuration file.

```yaml
defaults:
  - base_config
  - path_config/NAME_OF_NEW_PATH_CONFIG_FILE
```

### Scripts


Training script
```bash
python scripts/learning_scripts/train_test_simple_net.py
```


## Citation

```

```
