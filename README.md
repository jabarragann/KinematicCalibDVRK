# Kinematic error model of dVRK robot


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
