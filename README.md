# Image Feature Extractor

This repository provides a Python implementation to extract visual features from images, by means of pretrained convolutional neural networks (e.g., ResNet50). The extracted features correspond to the output of either fully-connected or convolutional layers.

This repository was adopted for the following papers:

- [A Study on the Relative Importance of Convolutional Neural Networks in Visually-Aware Recommender Systems](https://github.com/sisinflab/CNNs-in-VRSs)

**Table of Contents:**
- [Requirements](#requirements)
- [Extract features](#extract-features)
  - [Please notice](#please-notice)
  - [Available CNNs](#available-cnns)
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
### Please notice
The input parameters ```model_name```, ```cnn_output_name```, and ```cnn_output_shape``` are lists of values for whom there must exist a correspondence across all the lists, e.g., ```model_name[0] --> VGG19```, ```cnn_output_name[0] --> fc2```, ```cnn_output_shape[0] --> ()```. Setting the output shape as ```()``` means no reshape is performed after extraction. Currently, the implemented dimensionality reduction is just PCA.

### Available CNNs
- AlexNet ([PyTorch](https://pytorch.org/hub/pytorch_vision_alexnet/))
- VGG19 ([Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG19))
- ResNet50 ([Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50))
- ResNet152 ([Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152))

### Outputs
The script will generate three output files, namely:
- ```classes_<model_name>.csv```, a csv file with the classification outcomes for the input images and the adopted model
- ```cnn_features_<model_name>_<output_name>.npy```, a npy file with the extracted features for the input images, the adopted model and extraction layer
- ```category_features_<model_name>_<output_name>_pca<category_dim>.npy```, a npy file with the extracted features for the input images, the adopted model and extraction layer, and reduction dimension.
