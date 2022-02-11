import os
import argparse
import csv
import json

from operator import itemgetter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature extraction for words in text.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to run experiments')
    parser.add_argument('--dataset', nargs='?', default='amazon_baby', help='dataset path')
    parser.add_argument('--max_tokens', type=int, default=100, help='max number of tokens')
    parser.add_argument('--model_name', nargs='+', type=str, default=['word2vec-google-news-300'],
                        help='model for feature extraction')
    parser.add_argument('--normalize', type=bool, default=True, help='whether to normalize output or not')
    parser.add_argument('--print_each', type=int, default=100, help='print each n samples')

    return parser.parse_args()


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

from config.configs import *
from utils.write import *
from utils.read import *

import gensim.downloader
import numpy as np

import time
import sys


def find_indices_vocabulary(tokens, voc):
    if ' ' not in tokens:
        return [voc[tokens]]
    else:
        return list(itemgetter(*tokens)(voc))


def extract():
    for id_model, m in enumerate(args.model_name):
        print('****************************************************************')
        print('EXTRACTION MODEL: %s' % m)

        # model setting
        word2vec_model = gensim.downloader.load(args.model_name[id_model])

        # dataset padding
        data = read_csv(reviews_path.format(args.dataset), sep='\t')
        print('Loaded dataset from %s' % reviews_path.format(args.dataset))

        vocabulary_users = set()
        vocabulary_items = set()
        for u in data['USER_ID'].unique().tolist():
            list_of_tokens = ' '.join(data[data['USER_ID'] == u]['tokens']).split(' ')
            for t in list_of_tokens:
                vocabulary_users.add(t)

        for i in data['ITEM_ID'].unique().tolist():
            list_of_tokens = ' '.join(data[data['ITEM_ID'] == i]['tokens']).split(' ')
            for t in list_of_tokens:
                vocabulary_items.add(t)

        vocabulary_users.add('<pad>')
        padding_index_users = len(vocabulary_users) - 1
        vocabulary_dict_users = {k: i for i, k in enumerate(list(vocabulary_users))}
        del vocabulary_users

        vocabulary_items.add('<pad>')
        padding_index_items = len(vocabulary_items) - 1
        vocabulary_dict_items = {k: i for i, k in enumerate(list(vocabulary_items))}
        del vocabulary_items

        users_tokens = {}
        items_tokens = {}
        for u in data['USER_ID'].unique().tolist():
            list_of_tokens = ' '.join(data[data['USER_ID'] == u]['tokens']).split(' ')
            list_of_tokens = find_indices_vocabulary(list_of_tokens, vocabulary_dict_users)
            if len(list_of_tokens) > args.max_tokens:
                list_of_tokens_padded = list_of_tokens[:args.max_tokens]
            else:
                list_of_tokens_padded = list_of_tokens + (
                            [padding_index_users] * (args.max_tokens - len(list_of_tokens)))
            users_tokens[str(u)] = list_of_tokens_padded

        for i in data['ITEM_ID'].unique().tolist():
            list_of_tokens = ' '.join(data[data['ITEM_ID'] == i]['tokens']).split(' ')
            list_of_tokens = find_indices_vocabulary(list_of_tokens, vocabulary_dict_items)
            if len(list_of_tokens) > args.max_tokens:
                list_of_tokens_padded = list_of_tokens[:args.max_tokens]
            else:
                list_of_tokens_padded = list_of_tokens + (
                        [padding_index_items] * (args.max_tokens - len(list_of_tokens)))
            items_tokens[str(i)] = list_of_tokens_padded

        users_filename = 'users_tokens_concat.json'
        with open('../data/{0}/{1}'.format(args.dataset, users_filename), 'w') as f:
            json.dump(users_tokens, f)

        items_filename = 'items_tokens_concat.json'
        with open('../data/{0}/{1}'.format(args.dataset, items_filename), 'w') as f:
            json.dump(items_tokens, f)
        del data, users_tokens, items_tokens

        # text words features
        text_words_features_vocabulary_users = np.zeros(
            shape=[len(list(vocabulary_dict_users.keys())), word2vec_model.vector_size]
        )
        text_words_features_vocabulary_items = np.zeros(
            shape=[len(list(vocabulary_dict_items.keys())), word2vec_model.vector_size]
        )

        # features extraction
        print('Starting vocabulary embedding extraction...\n')
        start = time.time()

        for v, idx in vocabulary_dict_users.items():
            try:
                text_words_features_vocabulary_users[idx] = word2vec_model.get_vector(v, norm=True)
            except KeyError:
                pass

            if (idx + 1) % args.print_each == 0:
                sys.stdout.write(
                    '\r%d/%d samples completed for users' % (idx + 1, len(list(vocabulary_dict_users.keys()))))
                sys.stdout.flush()

        for v, idx in vocabulary_dict_items.items():
            try:
                text_words_features_vocabulary_items[idx] = word2vec_model.get_vector(v, norm=True)
            except KeyError:
                pass

            if (idx + 1) % args.print_each == 0:
                sys.stdout.write(
                    '\r%d/%d samples completed for items' % (idx + 1, len(list(vocabulary_dict_items.keys()))))
                sys.stdout.flush()

        end = time.time()
        print('\n\nFeature extraction completed in %f seconds.' % (end - start))

        if args.normalize:
            text_words_features_vocabulary_users = text_words_features_vocabulary_users / np.max(
                np.abs(text_words_features_vocabulary_users))
            text_words_features_vocabulary_items = text_words_features_vocabulary_items / np.max(
                np.abs(text_words_features_vocabulary_items))

        save_np(npy=text_words_features_vocabulary_users,
                filename='../data/{0}/original/users_{1}.npy'.format(args.dataset, m.lower()))
        save_np(npy=text_words_features_vocabulary_items,
                filename='../data/{0}/original/items_{1}.npy'.format(args.dataset, m.lower()))


if __name__ == '__main__':
    extract()
