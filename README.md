# Image Feature Extractor

This repository provides a Python implementation to extract visual features from images, by means of pretrained convolutional neural networks (e.g., ResNet50). The extracted features correspond to the output of either fully-connected or convolutional layers.

This repository was adopted for the following papers:

- [A Study on the Relative Importance of Convolutional Neural Networks in Visually-Aware Recommender Systems (under review)](https://github.com/sisinflab/CNNs-in-VRSs)

**Table of Contents:**
- [Requirements](#requirements)
- [Extract features](#extract-features)
  - [Please notice](#please-notice)
  - [Available CNNs](#available-cnns)
  - [Available dimensionality reductions](#available-dimensionality-reductions)
  - [Outputs](#outputs)
- [Evaluate visual recommendations](#evaluate-visual-recommendations)
  - [Expected inputs](#expected-inputs)
  - [Outputs](#outputs-1)

## Requirements

To begin with, please make sure your system has these installed:

* Python 3.6.8
* CUDA 10.1
* cuDNN 7.6.4

Then, install all required Python dependencies with the command:
```
pip install -r requirements.txt
```
Finally, you are supposed to structure the dataset folders in the following way:
```
./data
  amazon_baby_vgg19/
    original/
       images/
        0.jpg
        1.jpg
        ...
  amazon_boys_girls_alexnet/
    original/
      images/
        0.jpg
        1.jpg
        ...
```
**N.B.** The dataset folder structure requires the notation ```<dataset_name>_<cnn_name>```, even though the different dataset folders contain the exact same files. This is due to the fact that, when training and evaluating state-of-the-art visual-based recommender systems on these datasets through [Elliot](https://github.com/sisinflab/elliot), they need to be recognized as different datasets.

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
The input parameters ```model_name```, ```cnn_output_name```, and ```cnn_output_shape``` are lists of values for whom there must exist a correspondence across all the lists, e.g., ```model_name[0] --> VGG19```, ```cnn_output_name[0] --> fc2```, ```cnn_output_shape[0] --> ()```. Setting the output shape as ```()``` means no reshape is performed after extraction.

### Available CNNs
- AlexNet ([PyTorch](https://pytorch.org/hub/pytorch_vision_alexnet/))
- VGG19 ([Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG19))
- ResNet50 ([Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50))
- ResNet152 ([Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152))

### Available dimensionality reductions
- Principal Component Analysis (PCA)

### Outputs
The script will generate three output files, namely:
- ```classes_<model_name>.csv```, a csv file with the classification outcomes for the input images and the adopted model
- ```cnn_features_<model_name>_<output_name>.npy```, a npy file with the extracted features for the input images, the adopted model and extraction layer
- ```category_features_<model_name>_<output_name>_pca<category_dim>.npy```, a npy file with the extracted features for the input images, the adopted model and extraction layer, and reduction dimension.

## Evaluate visual recommendations
This section refers to the novel metric *visual diversity* (**VisDiv**), proposed in our paper [A Study on the Relative Importance of Convolutional Neural Networks in Visually-Aware Recommender Systems](https://github.com/sisinflab/CNNs-in-VRSs). 

To calculate the **VisDiv**, please run the following script:
```
python evaluate_visual_profile.py \
  --dataset <dataset-name> \
  --image_feat_extractors <list-of-image-feature-extractors> \
  --visual_recommenders <list-of-visual-recommenders> \
  --top_k <top-k-to-calculate-visdiv-on> \
  --save_plots <whether-to-save-the-output-plots>
```

### Expected inputs
To run, the script requires the folder with the obtained recommendation results. It must be formatted in the following way:
```
./results/
  amazon_baby_vgg19/
    VBPR.tsv
    DeepStyle.tsv
    ...
  amazon_boys_girls_resnet50/
    ACF.tsv
    VNPR.tsv
    ...
```
where each tsv file refers to the recommendation lists produced by the best performing configuration for each visual recommender.

### Outputs
The script will generate the following outputs, namely:
- ```./plots/<dataset-name>_<top-k>/<visual-recommender>/<image-feature-extractor>/u_<user-id>.pdf```, a set of pdf files having the t-SNE graphical representation of the **VisDiv** for each user
- ```./plots/<dataset-name>_<top-k>/<visual-recommender>/<image-feature-extractor>/all_users_stats.csv```, a csv file to store all **VisDiv** values for each user
- ```./plots/<dataset-name>_<top-k>/<visual-recommender>/<image-feature-extractor>/final_stats.out```, a txt file to store the final statistics for the **VisDiv** metric
