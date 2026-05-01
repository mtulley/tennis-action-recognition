# Tennis Shot Action Recognition with NVIDIA TAO and Deepstream


## Overview
This repository contains deep learning models trained for classification of tennis shots from input videos.

## Project Summary

## Codebase Map

## Architecture Diagrams
<p align="center">
  <img src="images/Architecture_AR_1.png" width="45%"/>
  <img src="images/Architecture_AR_2.png" width="45%"/>
</p>
<p align="center">
  <em>Model architecture diagrams</em>
</p>
## Setup instructions
For Action Recognition net, install tao tool kit via docker using a legacy API key, run training with the desired YAML file , if needed the pretrained weights used in this project can also be donwloaded from nvidias NGC website.


## Repo Structure


```
.
├── dataset/                    # Empty directory for dataset
├── notebooks/                  # Contains a notebook for testing
├── pretrained/                 # Empty directory for pretrained model weights
├── results/                    # Results of training/evaluation/testing
│   ├── pose_classification/    
│   └── action_recognition/     
├── scripts/                    # Python scripts
│   ├── pose_classification/    
│   └── action_recognition/     
├── sources/                    # Deepstream app source files
│   ├── pose_classification/    
│   └── action_recognition/     
├── specs/                      # Contains .yaml specification files for TAO
│   ├── pose_classification/
│   └── action_recognition/
├── .gitignore                    
└── README.md
```


## Dataset
The data is from the THETIS dataset (link here).

### Videos
- 640x480p
- 55 people
- 3 videos per person per shot
- 55 peoples x 3 videos x 12 shots = 1980 videos

### Skeletons
- 34 joints (Nvidia format for PoseClassificationNet)
- Keypoint information stored in json files

