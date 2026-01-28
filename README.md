# Skeleton2Stage

Official implementation of paper: "Skeleton2Stage: Reward-Guided Fine-Tuning for Physically Plausible Dance Generation". Prior dance generation methods often operate on sparse skeletons and overlook geometric constraints of the human body, leading to artifacts such as penetration and foot sliding. We propose a reward-guided fine-tuning framework that aligns generated motions with body geometry, improving physical plausibility.

# Table of Contents
* [News 🚩](#news-)
* [TODOs](#todos)
* [Introduction](#introduction)
    * [Docs](#docs)
    * [Current Results on EDGE](#current-results-on-edge)
    * [Dependencies](#dependencies)
* [Evaluation](#evaluation)
    * [Evaluation Pipeline](#evaluation-pipeline)
    * [Evaluation on Other Models](#evaluation-on-different-model)
* [Training](#training)
    * [Data Processing](#data-processing)
    * [Training Imitation Policy](#training-imitation-policy)
    * [RLFT for EDGE](#rlft-for-edge)
    * [RLFT for Other Models](#rlft-for-other-models)
* [Rendering](#rendering)
* [Trouble Shooting](#trouble-shooting)
    * [Supporting GPU types newer than A100](#supporting-gpu-types-newer-than-a100)
* [Citation](#citation)
* [References](#references)

## News 🚩


## TODOs

- [] Supports for training on GPU newer than A100.

- [] Release training imitation policy code.

- [] Release training code. 

- [] Release evaluation code. 

- [] Release rendering code.

## Introduction


### Docs 

### Current Results on EDGE

All evaluation is done using the mean SMPL body shape.

### Dependencies

To create the environment, follow the following instructions: 

1. Create new conda environment and install pytroch:


```
conda create -n isaac python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install -r requirement.txt
```

2. Download and setup [Isaac Gym](https://developer.nvidia.com/isaac-gym). 


3. Download SMPL paramters from [SMPL](https://smpl.is.tue.mpg.de/) and [SMPLX](https://smpl-x.is.tue.mpg.de/download.php). Put them in the `data/smpl` folder, unzip them into 'data/smpl' folder. For SMPL, please download the v1.1.0 version, which contains the neutral humanoid. Rename the files `basicmodel_neutral_lbs_10_207_0_v1.1.0`, `basicmodel_m_lbs_10_207_0_v1.1.0.pkl`, `basicmodel_f_lbs_10_207_0_v1.1.0.pkl` to `SMPL_NEUTRAL.pkl`, `SMPL_MALE.pkl` and `SMPL_FEMALE.pkl`. For SMPLX, please download the v1.1 version. Rename The file structure should look like this:

```

|-- data
    |-- smpl
        |-- SMPL_FEMALE.pkl
        |-- SMPL_NEUTRAL.pkl
        |-- SMPL_MALE.pkl
        |-- SMPLX_FEMALE.pkl
        |-- SMPLX_NEUTRAL.pkl
        |-- SMPLX_MALE.pkl

```


4. Use the following script to download trained models and sample data.

```
bash download_data.sh
```

this will download amass_isaac_standing_upright_slim.pkl, which is a standing still pose for testing. 

To evaluate with your own SMPL data, see the script `scripts/data_process/convert_data_smpl.py`. Pay speical attention to make sure the coordinate system is the same as the one used  in simulaiton (with negative z as gravity direction). 

Make sure you have the SMPL paramters properly setup by running the following scripts:
```
python scripts/vis/vis_motion_mj.py
python scripts/joint_monkey_smpl.py
```

The SMPL model is used to adjust the height the humanoid robot to avoid penetnration with the ground during data loading. 


## Evaluation 

### Evaluation Pipeline

### Evaluation on Other Models


## Training

### Data Processing

### Training Imitation Policy

### RLFT for EDGE

### RLFT for Other Models


## Rendering


## Trouble Shooting

### Supporting GPU types newer than A100


## Citation
If you find this work useful for your research, please cite our paper:
```
   
```

## References
This repository is built on top of the following amazing repositories:
* Main code framework is from: [EDGE](https://github.com/Stanford-TML/EDGE)
* Imitation policy is from: [vid2player3d](https://github.com/nv-tlabs/vid2player3d)
* SMPL models and layer is from: [SMPL-X model](https://github.com/vchoutas/smplx)
* README template is from: [PHC](https://github.com/ZhengyiLuo/PHC)

Please follow the lisence of the above repositories for usage. 