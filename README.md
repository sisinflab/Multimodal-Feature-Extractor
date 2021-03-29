# Image Feature Extractor

This repository provides a Python implementation to extract visual features from images, by means of pretrained convolutional neural networks (e.g., ResNet50). The extracted features correspond to the output of either fully-connected or convolutional layers. This repository was exploited in the following papers:

- [A Study on the Relative Importance of Convolutional Neural Networks in Visually-Aware Recommender Systems](https://github.com/sisinflab/CNNs-in-VRSs)

**Table of Contents:**
- [Requirements](#requirements)
- [Extract features](#extract-features)
- [Evaluate visual recommendations](#evaluate-visual-recommendations)

## Requirements

To begin with, please make sure your system has these installed:

* Python 3.6.8
* CUDA 10.1
* cuDNN 7.6.4

Then, install all required Python dependencies with the command:
```
pip install -r requirements.txt
```

## Extract features
