import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature extraction for original texts.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to run experiments')
    parser.add_argument('--dataset', nargs='?', default='amazon_boys_girls', help='dataset path')
    parser.add_argument('--model_name', nargs='+', type=str, default=['all-mpnet-base-v2'],
                        help='model for feature extraction')
    parser.add_argument('--text_output_split', nargs='+', type=bool, default=[True],
                        help='whether output should be split')
    parser.add_argument('--normalize', type=bool, default=True, help='whether to normalize output or not')
    parser.add_argument('--column', nargs='?', default='REVIEW', help='column of the dataframe to encode')
    parser.add_argument('--input_file', nargs='?', default='train_reviews', help='input filename')
    parser.add_argument('--print_each', type=int, default=100, help='print each n samples')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from config.configs import *
from utils.write import *
from utils.read import *
from sentence_transformers import SentenceTransformer

import numpy as np
import time
import sys


def extract():
    for id_model, m in enumerate(args.model_name):
        print('****************************************************************')
        print('EXTRACTION MODEL: %s' % m)

        if args.text_output_split[id_model]:
            # create directories for split
            if not os.path.exists('../data/{0}/original/{1}_{2}'.format(args.dataset, args.input_file, m.lower())):
                os.makedirs('../data/{0}/original/{1}_{2}'.format(args.dataset, args.input_file, m.lower()))

        # model setting
        text_model = SentenceTransformer(args.model_name[id_model])

        # dataset setting
        data = read_csv('../data/{0}/original/{1}.tsv'.format(args.dataset, args.input_file), sep='\t')
        print('Loaded dataset from %s' % descriptions_path.format(args.dataset))

        # text features
        text_features = np.empty(
            shape=[len(data), text_model.get_sentence_embedding_dimension()]
        )

        # features extraction
        print('Starting extraction...\n')
        start = time.time()

        for index, row in data.iterrows():
            # text features extraction
            text_features[index] = text_model.encode(sentences=str(row[args.column]))

            if (index + 1) % args.print_each == 0:
                sys.stdout.write('\r%d/%d samples completed' % (index + 1, len(data)))
                sys.stdout.flush()

        end = time.time()
        print('\n\nFeature extraction completed in %f seconds.' % (end - start))

        if args.normalize:
            text_features = text_features / np.max(np.abs(text_features))

        if args.text_output_split[id_model]:
            for d in range(len(data)):
                save_np(npy=text_features[d],
                        filename='../data/{0}/original/{1}_{2}'.format(args.dataset, args.input_file, m.lower()) + str(d) + '.npy')
            print('Saved text features numpy to ==> %s' %
                  text_features_dir.format(args.dataset, m.lower()))
        else:
            save_np(npy=text_features,
                    filename='../data/{0}/original/{1}_{2}.npy'.format(args.dataset, args.input_file, m.lower()))
            print('Saved text features numpy to ==> %s' %
                  '../data/{0}/original/{1}_{2}.npy'.format(args.dataset, args.input_file, m.lower()))


if __name__ == '__main__':
    extract()
