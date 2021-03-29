import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# VBPR-like:
# - AlexNet (5)
# - ResNet50 (avg_pool)
# - VGG19 (fc2)
# ACF:
# - AlexNet (0 --> None) --> (36, 256)
# - ResNet50 (conv5_block3_out) --> (49, 2048)
# - VGG19 (block5_pool) --> (49, 512)


def parse_args():
    parser = argparse.ArgumentParser(description="Run classification and feature extraction for original images.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to run experiments')
    parser.add_argument('--dataset', nargs='?', default='amazon_boys_girls_reduced', help='dataset path')
    parser.add_argument('--model_name', nargs='+', type=str, default=['AlexNet', 'ResNet50', 'VGG19'], help='model for feature extraction')
    parser.add_argument('--cnn_output_name', nargs='+', default=[0, 'conv5_block3_out', 'block5_pool'], help='output layer name')
    parser.add_argument('--cnn_output_shape', nargs='+', type=tuple, default=[(36, 256), (49, 2048), (49, 512)], help='output shape for cnn output (e.g., ACF)')
    parser.add_argument('--category_dim', type=int, default=128, help='dimensionality reduction for category')
    parser.add_argument('--print_each', type=int, default=100, help='print each n samples')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from vision.CnnFeatureExtractor import *
from vision.Dataset import *
from config.configs import *
from utils.write import *
from utils.read import *

import numpy as np
import time
import sys
import csv


def classify_extract():
    for id_model, m in enumerate(args.model_name):
        print('*****************************************************************')
        print('EXTRACTION MODEL: %s' % m)
        print('OUTPUT LAYER: %s' % args.cnn_output_name[id_model])
        # model setting
        cnn_model = CnnFeatureExtractor(args.gpu,
                                        read_imagenet_classes_txt(imagenet_classes_path),
                                        m,
                                        args.cnn_output_name[id_model],
                                        args.category_dim,
                                        args.cnn_output_shape[id_model])

        # dataset setting
        data = Dataset(
            dataset=args.dataset,
            resize=(224, 224),
            model_name=m
        )
        print('Loaded dataset from %s' % images_path.format(args.dataset))

        # image features
        if m == 'AlexNet':
            cnn_features = np.empty(shape=[data.num_samples, *cnn_model.get_out_shape()])
        else:
            cnn_features = np.empty(shape=[data.num_samples, *cnn_model.model.get_layer(args.cnn_output_name[id_model]).output.shape[1:]])

        # classification and features extraction
        print('Starting classification...\n')
        start = time.time()

        with open(classes_path.format(args.dataset, m.lower()), 'w') as f:
            fieldnames = ['ImageID', 'ClassStr', 'ClassNum', 'Prob']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for i, d in enumerate(data):
                norm_image, original_image, path = d

                # classification
                out_class = cnn_model.classify(sample=(norm_image, path))
                writer.writerow(out_class)

                # image features extraction
                cnn_features[i] = cnn_model.extract_feature(image=norm_image)

                if (i + 1) % args.print_each == 0:
                    sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
                    sys.stdout.flush()

        # if cnn features must be reshaped
        if len(args.cnn_output_shape[id_model]):
            categories_reshaped = cnn_features.reshape((data.num_samples, *args.cnn_output_shape[id_model]))
            save_np(npy=categories_reshaped,
                    filename=cnn_features_path.format(args.dataset,
                                                      m.lower(),
                                                      str(args.cnn_output_name[id_model])))
        # if no reshape is applied to cnn features
        else:
            save_np(npy=cnn_features,
                    filename=cnn_features_path.format(args.dataset,
                                                      m.lower(),
                                                      str(args.cnn_output_name[id_model])))

            categories_pca = cnn_model.pca_reduction(cnn_features)
            save_np(npy=categories_pca,
                    filename=category_features_path.format(args.dataset,
                                                           m.lower(),
                                                           str(args.cnn_output_name[id_model]),
                                                           args.category_dim))

        end = time.time()

        print('\n\nClassification and feature extraction completed in %f seconds.' % (end - start))
        print('Saved cnn features numpy to ==> %s' %
              cnn_features_path.format(args.dataset, m.lower(), args.cnn_output_name[id_model]))
        print('Saved pca reduced features numpy to ==> %s' %
              category_features_path.format(args.dataset, m.lower(), args.cnn_output_name[id_model], args.category_dim))
        print('Saved classification file to ==> %s' % classes_path.format(args.dataset, m.lower()))


if __name__ == '__main__':
    classify_extract()
