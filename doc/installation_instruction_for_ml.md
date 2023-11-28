# Installation instructions for ML section of the code

## Import packages
hydra-core==1.3.2
torch==2.1.0+cu118
numpy==1.26.0

## Installation on anaconda env

1. remove ros dependencies from requirements.txt
2. Install surgical robotics client to env. Only fk and ik funcs will be used so hopefully no ros dependencies will be needed.
3. Install pytorch. My env uses `torch==2.1.0+cu118`