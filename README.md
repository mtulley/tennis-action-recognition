# Tennis Shot Action Recognition with NVIDIA TAO and Deepstream


## Overview
This repository contains deep learning models trained for classification of tennis shots from input videos.

## Project Summary
This project explores the use of NVIDIA TAO Toolkit and DeepStream pipelines to perform tennis action recognition from video using lightweight deep learning models. Two approaches are evaluated: an end-to-end 3D CNN (ActionRecognitionNet) and a skeleton-based pipeline (BodyPose3DNet + PoseClassificationNet). While the 3D CNN achieves strong performance (over 90% accuracy) on the THETIS dataset, the skeleton-based method performs significantly worse, highlighting the importance of visual context such as racket motion. Despite promising training results, real-time inference on live video is less reliable due to dataset limitations and poor generalization, suggesting that model performance is constrained more by data quality than model capacity.

## Architecture Diagrams
<p align="center">
  <img src="images/Architecture_AR_1.png" width="45%"/>
  <img src="images/Architecture_AR_2.png" width="45%"/>
</p>
<p align="center">
  <em>Action Recognition Net 3D architecture diagrams</em>
</p>

## Setup instructions
For Action Recognition net, install tao tool kit via docker using a legacy API key, run training with the desired YAML file , if needed the pretrained weights used in this project can also be donwloaded from nvidias NGC website.


## CodeBase Map/Repo Structure


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

## Results summary
<p align="center">
  <img src="images/data1.png"/>
</p>
<p align="center">

The experiments show a clear progression in performance driven by key parameter changes. The initial baseline using pretrained weights struggled with certain classes, achieving only around 50% accuracy in some cases. A major improvement came from reducing the RGB sequence length and adding dropout, which increased overall accuracy by about 12% and revealed that training plateaued relatively early. Further experiments confirmed that a short sequence length was optimal, and additional gains were achieved by lowering the learning rate, reducing batch size, and continuing training from a strong checkpoint, which helped push performance beyond the previous ~85% ceiling. Final tuning with short training intervals, slightly higher learning rates, and increased decay allowed the model to surpass 90% accuracy, though with more unstable training behavior compared to earlier runs.

## AI usage summary
LLM and AI usage was primarily used to deal with issues when installing tao tool kit, for example figuring out correct versions, what version of weights are compatible and should be donwloaded, the cli interface commands (as the official documentation was for a different version of toolkit). AI was also used to edit the c backend of deepstream config files, primarily for the purpose of adding a working video source and output source as the fakesink was incompatible. 

### Videos
- 640x480p
- 55 people
- 3 videos per person per shot
- 55 peoples x 3 videos x 12 shots = 1980 videos

### Skeletons
- 34 joints (Nvidia format for PoseClassificationNet)
- Keypoint information stored in json files



