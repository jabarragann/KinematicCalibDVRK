# Improving the realism of robotic surgery simulation through injection of learning-based estimated errors

## Setup environment - Last modified on August 01, 2024

The following code base requires a virtual environment to work. This code was tested with anaconda virtual environment.

```bash
conda create -n ki_err python=3.10 numpy
# The following command might install other pytorch version in the future. For now it is pytorch 2.4.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 
pip install -r requirements.txt #python requirements
pip install -r requirements_torch.txt # torch requirements
pip install -r requirements_ros.txt # install ros dependencies - PYKDL needs to be compiled usually so this will take some time ~ 2-5min
pip install -e . # install our package
```

Note: if ros_requirements fails, you install all the software in your main python interpreter.

Install old version of surgical robotics challenge. This code base uses the FK and IK of the SRC. I will try to updated to the new version soon. For now unfortunately you need to install my fork.
```bash
git clone https://github.com/jabarragann/surgical_robotics_challenge.git
git checkout -b fk_patch origin/fk_patch
```

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
