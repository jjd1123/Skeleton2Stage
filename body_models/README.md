# Body Models Setup Guide

This directory is a placeholder for the SMPL, SMPL-H, and SMPL-X body models, which may be required to run the code. Due to licensing restrictions, these models cannot be distributed directly in this repository.

Please follow the steps below to download and set up the models correctly.

## Download the Models

You must register and download the models from their official websites, and you should also run `bash download_smpl_files.sh` if you want to use MDM related features (MDM+physics-based motion projection).

## Expected Directory Structure

After downloading and extracting the models, please organize them to match the exact structure shown below. The code expects to find the files in these specific paths.

```bash
body_models/
├── README.md            # This guide file
│
├── smpl/
│   ├── J_regressor_extra.npy
│   ├── kintree_table.pkl
│   ├── smplfaces.npy
│   ├── SMPL_FEMALE.pkl
│   ├── SMPL_MALE.pkl
│   └── SMPL_NEUTRAL.pkl
│
├── smplh/
│   ├── LICENSE.txt
│   ├── female/
│   │   └── model.npz
│   ├── male/
│   │   └── model.npz
│   └── neutral/
│       └── model.npz
│
└── smplx/
    ├── female/
    │   └── model.npz
    ├── male/
    │   └── model.npz
    └── neutral/
        └── model.npz
```