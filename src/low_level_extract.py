from vision.LowFeatureExtractor import *
from vision.Dataset import *
from config.configs import *
from utils.write import *

import numpy as np
import time
import sys
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run low-level feature extraction for original images.")
    parser.add_argument('--dataset', nargs='?', default='amazon_men', help='dataset path')
    parser.add_argument('--num_bins', type=int, default=8, help='number of bins for color histogram')
    parser.add_argument('--low_level_output_split', type=bool, default=True, help='whether output should be split')
    parser.add_argument('--print_each', type=int, default=100, help='print each n samples')

    return parser.parse_args()


def low_level_extract():
    args = parse_args()

    if not os.path.exists(shape_features_dir.format(args.dataset)):
        os.makedirs(shape_features_dir.format(args.dataset))

    if not os.path.exists(color_features_dir.format(args.dataset, args.num_bins)):
        os.makedirs(color_features_dir.format(args.dataset, args.num_bins))

    low_level_model = LowFeatureExtractor(args)

    # dataset setting
    data = Dataset(
        dataset=args.dataset,
        model_name=None,
        resize=None,
        normalize=False
    )
    print('Loaded dataset from %s' % images_path.format(args.dataset))

    # image features
    colors = np.empty(shape=[data.num_samples, args.num_bins * args.num_bins * args.num_bins])

    # low-level feature extraction
    print('Starting low-level feature extraction...\n')
    start = time.time()

    for i, d in enumerate(data):
        _, original_image, path = d

        # low-level feature extraction
        color, shape = low_level_model.extract_color_shape(sample=(original_image, path))
        colors[i] = color
        Image.fromarray(shape).save(shape_features_dir.format(args.dataset) + str(i) + '.tiff')

        if (i + 1) % args.print_each == 0:
            sys.stdout.write('\r%d/%d samples completed' % (i + 1, data.num_samples))
            sys.stdout.flush()

    end = time.time()

    print('\n\nLow-level feature extraction completed in %f seconds.' % (end - start))
    print('Saved shapes features numpy to ==> %s' % shape_features_dir.format(args.dataset))

    if args.low_level_output_split:
        for d in range(data.num_samples):
            save_np(npy=colors[d], filename=color_features_dir.format(args.dataset, args.num_bins) + str(d) + '.npy')
        print('Saved colors features numpy to ==> %s' % color_features_dir.format(args.dataset, args.num_bins))
    else:
        save_np(npy=colors, filename=color_features_path.format(args.dataset, args.num_bins))
        print('Saved colors features numpy to ==> %s' % color_features_path.format(args.dataset, args.num_bins))


if __name__ == '__main__':
    low_level_extract()
