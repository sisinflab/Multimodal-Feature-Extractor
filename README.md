# Image Feature Extractor

This repository provides a Python implementation to extract visual features from images, by means of pretrained convolutional neural networks (e.g., ResNet50). The extracted features correspond to the output of either fully-connected or convolutional layers.

This repository was adopted for the following papers:

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

To extract visual features from images, please run the following script:
```
python classify_extract.py \
  --gpu <gpu-id>
  --dataset <dataset-name> \
  --model_name <list-of-cnns> \
  --cnn_output_name <list-of-output-names-for-each-cnn> \
  --cnn_output_shape <list-of-output-shapes-for-each-cnn> \
  --category_dim <dimension-for-dimensionality-reduction> \
  --print_each <print-status-each>
```
**N.B.** the input parameters ```model_name```, ```cnn_output_name```, and ```cnn_output_shape``` are lists of values in which there must exist a correspondance across all the lists, e.g., ```model_name\[0\] --> VGG19```, ```cnn_output_name\[0\] --> fc2```, ```cnn_output_shape\[0\] --> ()```. 
