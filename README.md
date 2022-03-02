# Multimodal Feature Extractor

This repository provides a Python implementation to extract multimodal features from images and texts, either high-level ones from pretrained deep learning models (e.g., CNNs-extracted embeddings), or low-level ones (e.g., color and shape).

This repository was used in:

- [A Study on the Relative Importance of Convolutional Neural Networks in Visually-Aware Recommender Systems (accepted at CVFAD@CVPR2021)](https://github.com/sisinflab/CNNs-in-VRSs)
- [V-Elliot: Design, Evaluate and Tune Visual Recommender Systems (accepted at RecSys2021)](https://github.com/sisinflab/elliot)
- [Leveraging Content-Style Item Representation for Visual Recommendation (accepted at ECIR2022)](https://github.com/sisinflab/Content-Style-VRSs)

**Table of Contents:**
- [Requirements](#requirements)
- [Extract features](#extract-features)
  - [Visual features](#visual-features)
      - [Useful info](#useful-info)
      - [Available CNNs](#available-cnns)
      - [Available dimensionality reductions](#available-dimensionality-reductions)
      - [Outputs](#outputs)
  - [Textual features](#textual-features)
      - [Available textual encoders](#available-cnns)
      - [Outputs](#outputs-1)
- [Evaluate visual recommendations](#evaluate-visual-recommendations)
  - [Expected inputs](#expected-inputs)
  - [Outputs](#outputs-1)
- [Main Contact](#main-contact)

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
# EXAMPLE VISUAL DATA
./data
  amazon_baby/
    original/
       images/
        0.jpg
        1.jpg
        ...
  amazon_boys_girls/
    original/
      images/
        0.jpg
        1.jpg
        ...
        
# EXAMPLE TEXTUAL DATA
./data
  amazon_baby
    original/
      all_items_descriptions.tsv
  amazon_boys_girls/
    original/
      all_items_descriptions.tsv
```

## Extract features

### Visual features
To classify images and extract visual features from them, please run the following script:
```
python classify_extract_visual.py \
  --gpu <gpu-id>
  --dataset <dataset-name> \
  --model_name <list-of-cnns> \
  --cnn_output_name <list-of-output-names-for-each-cnn> \
  --cnn_output_shape <list-of-output-shapes-for-each-cnn> \
  --cnn_output_split <whether-to-store-separately-output-features-or-not> \
  --category_dim <dimension-for-dimensionality-reduction> \
  --print_each <print-status-each>
```
#### Useful info
The input parameters ```model_name```, ```cnn_output_name```, and ```cnn_output_shape``` are lists of values for whom there must exist a correspondence across all the lists, e.g., ```model_name[0] --> VGG19```, ```cnn_output_name[0] --> fc2```, ```cnn_output_shape[0] --> ()```. Setting the output shape as ```()``` means no reshape is performed after extraction.

#### Available CNNs
- AlexNet ([PyTorch](https://pytorch.org/hub/pytorch_vision_alexnet/))
- VGG19 ([Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG19))
- ResNet50 ([Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50))
- ResNet152 ([Keras](https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet152))

#### Available dimensionality reductions
- Principal Component Analysis (PCA)

#### Outputs
The script will generate three output files, namely:
- ```classes_<model_name>.csv```, a csv file with the classification outcomes for the input images and the adopted model
- ```cnn_features_<model_name>_<output_name>.npy```, a npy file with the extracted features for the input images, the adopted model and extraction layer
- ```cnn_features_<model_name>_<output_name>_pca<dim>.npy```, a npy file with the extracted features for the input images, the adopted model, extraction layer, and reduction dimension.

**N.B.** Depending on how you set the argument ```--cnn_output_split```, you may store a unique numpy array (see above), or different numpy arrays, one for each extracted visual feature (in this case, they will be stored to the directory ```cnn_features_<model_name>_<output_name>/``` or ```cnn_features_<model_name>_<output_name>_pca<dim>/```).

### Textual features
To extract textual features from texts, please run the following script:
```
python extract_textual.py \
  --gpu <gpu-id>
  --dataset <dataset-name> \
  --model_name <list-of-textual-encoders> \
  --text_output_split <whether-to-store-separately-output-features-or-not>
  --column <column-to-encode>
  --print_each <print-status-each>
```

#### Available textual encoders
Please, refer to [SentenceTransformers](https://www.sbert.net/) for an indication of the available pre-trained models.

#### Outputs
The script will generate three output files, namely:
- ```text_features_<model_name>.npy```, a npy file with the extracted features for the input texts and the adopted model

**N.B.** Depending on how you set the argument ```--text_output_split```, you may store a unique numpy array (see above), or different numpy arrays, one for each extracted textual feature (in this case, they will be stored to the directory ```text_features_<model_name>/```).

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

## Main Contact
Daniele Malitesta (daniele.malitesta@poliba.it)
