# DATASET
data_path = '../data/{0}/'
imagenet_classes_path = '../data/imagenet_classes.txt'
original = data_path + 'original/'
images_path = original + 'images/'
classes_path = original + 'classes_{1}.csv'

# FEATURES
cnn_features_path = original + 'cnn_features_{1}_{2}.npy'
cnn_features_dir = original + 'cnn_features_{1}_{2}/'
cnn_features_pca_path = original + 'cnn_features_{1}_{2}_pca{3}.npy'
cnn_features_pca_dir = original + 'cnn_features_{1}_{2}_pca{3}/'
