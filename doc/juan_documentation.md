# Improving the realism of robotic surgery simulation through injection of learning-based estimated errors

## Installation of crtk packages in virtual env
Create anaconda virtual env
```bash
conda create -n ki_err python=3.10 ipython -y
```

Localize installation of ros packages (sourcing ros_ws)

```bash
python -c "import crtk; print(crtk.__file__)"
```

Localize installation of virtual env (after sourcing virtual env)
```bash
conda info --envs | grep "\*" | awk '{print $3}'
```

Add `.pth` file to site-packages folder
```bash
echo "/home/jbarrag3/research_juan/ros_ws/dvrk_ws/devel/lib/python3/dist-packages" > /home/jbarrag3/anaconda3/envs/ki_err/lib/python3.10/site-packages/dvrk_crtk.pth
```

## Hardware installation

1. Generate a ros worskpace with dvrk and atracsys repositories
2. Compile workspace
3. The previous compiled should have ignored the atracsys wrapper. Now add the sdk and recompile as described [here](https://github.com/jhu-saw/sawAtracsysFusionTrack/tree/devel)
4. Recompile ros worspace with `catkin build --force-cmake --summary`

## Scripts usage

**Neural network training**

Neural net training script uses hydra to manage configurations. Default params are defined in config folder. When running the script, configurations can be overwritten with they syntax `config_name.param_name=value`. 

```bash
python ./scripts/learning_scripts/train_test_simple_net.py test_config.batch_size=64 train_config.batch_size=64 train_config.epochs=200
```

# TODO

**Each record is responsible for managing None data**

**Register to atracsys registration error**

List of atracsys topics
```bash
/atracsys/Controller/measured_cp_array # geometry_msgs/PoseArray 
/atracsys/dvrk_tip_frame/measured_cp #Type: geometry_msgs/PoseStamped
/atracsys/dvrk_tip_frame/registration_error #Type: std_msgs/Float64
```

**Errors in rostopics callback are not be printed in the terminal**

**Robotics toolbox changes matplotlib figures**
```
import roboticstoolbox
```