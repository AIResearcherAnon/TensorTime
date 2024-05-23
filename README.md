# TensorTime

Prerequisites

To set up the environment for running the code, follow these steps:

1. All experiments were conducted using the Docker image pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel.
2. First, install the required Python packages using requirements.txt: pip install -r requirements.txt
3.	Then, update and install the necessary system dependencies:
        apt-get update -y
        apt-get install -y libgl1-mesa-glx
        apt-get install -y libglib2.0-0

Training

1.	Configure the training settings in config.py and config_pathoptimization.py.
2.	To train the stochastic interpolant with tensor-valued time, run: python train.py
3.	To perform path optimization with a model trained using tensor-valued time, run: python train_pathoptimization.py

Evaluation

1.	To evaluate a stochastic interpolant model trained with tensor-valued time, generate samples using: python generate_samples.py. Then, calculate the FID using: python calculate_fid.py
2.	To evaluate the results of path optimization, generate samples using: python generate_samples_for_pathoptimization.py. Then, calculate the FID using: python calculate_fid.py
