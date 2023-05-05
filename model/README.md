# Siamese Network for image representation learning

Goal:
* Help MCL problem by finding images that close to each other instead of compare difference pixel-by-pixel.

## Table of Contents
1. [Setup](#setup)
1. [Files and Dirs](#files-and-dirs)
1. [Training](#training)
1. [Data description](#data-description)
1. [Folder Structure](#folder-structure)
---

## Setup
1. Install extra libraries required for this task (after installing required libraries in main directory):
    ```bash
    pip install -r requirements.txt
    ```

<p align="right"><a href="#siamese-network-for-image-representation-learning">[Back to top]</a></p>

---

## Files and Dirs
* `/config/`: contain `configuration.json` that defines the model
* `/data/`:
    * `csv-generator.py`: generate `metadata.csv`
    * Data is construct in folders demonstrated as in [Folder Structure](#folder-structure) section where every folder is a time that Cozmo bot rotate 360 degrees and take images.
* `/dataset/`: contains `dataset.py`, a dataset class for this model
* `/model/`: contains `model.py` that has our custom model class
* `/saved/`: contained weight of trained models
* `/utils/`: contains functions used in training and inference
* `infer.py`: contains functions to infer trained model
* `train.py`: contains functions to train model

<p align="right"><a href="#siamese-network-for-image-representation-learning">[Back to top]</a></p>

---

## Training
1. In `/config/configuration.json`, change model config depending on your need:
    * *model_type*: type of backbone in `torchvision.models`. By default, backbone is **resnet18**, or **vgg16_bn** otherwise.
    * *emb_size*: embedding size of the image (model's output). Default is 512.
1. Run command:
    ```bash
    python train.py
    ```

<p align="right"><a href="#siamese-network-for-image-representation-learning">[Back to top]</a></p>

---

## Data description
* The data was collected for Cozmo bot project presentation in Glat112, so all images were taken in either Glat112 (computer lab) or Glat207 (student lounge) at Gettysburg College.
* There are a total of 1440 images (taken 20 times) in jpg format. Each time taken, the Cozmo bot rotated 360 degrees and take an image every 5 degrees ($360/5=72$ images/time)
* Image is a gray-scaled 240x320 image taken using Cozmo bot built-in camera with face recognition algorithm integrated.

<p align="right"><a href="#siamese-network-for-image-representation-learning">[Back to top]</a></p>

---

## Folder Structure
```bash
.
├───config
│   └───configuration.json
├───data
│   ├───csv-generator.py
│   ├───metadata.csv
│   ├───0
│   │   ├───0.jpg
│   │   ├───...
│   │   └───71.jpg
│   ├───1
│   │   ├───0.jpg
│   │   ├───...
│   │   └───71.jpg
│   ├───...
│   └───19
│       ├───0.jpg
│       ├───...
│       └───71.jpg
├───dataset
│   └───dataset.py
├───model
│   └───model.py
├───saved
│   ├───best_resnet18_056.pt
│   ├───best_resnet18_069_expand.pt
│   └───best_resnet18.pt
├───utils
│   └───util.py
├───infer.py
├───README.md
├───requirements.txt
└───train.py
```
<p align="right"><a href="#siamese-network-for-image-representation-learning">[Back to top]</a></p>
