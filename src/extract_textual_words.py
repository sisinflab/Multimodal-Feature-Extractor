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
    parser.add_argument('--concat_tokens', type=bool, default=False, help='whether to concatenate tokens or not')
    parser.add_argument('--model_name', nargs='+', type=str, default=['word2vec-google-news-300'],
                        help='model for feature extraction')
    parser.add_argument('--text_output_split', nargs='+', type=bool, default=[False],
                        help='whether output should be split')
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

vocabulary = []


# def pad(tokens, max_number):
#     tokens_list = tokens.split(' ')
#     for t in tokens_list:
#         vocabulary.append(t)
#     if len(tokens_list) < max_number:
#         tokens_list += (['<pad>'] * (max_number - len(tokens_list)))
#     return tokens_list


def create_vocabulary(tokens):
    tokens_list = tokens.split(' ')
    for t in tokens_list:
        vocabulary.append(t)
    return len(tokens_list)


def find_indices_vocabulary(tokens, voc):
    if ' ' not in tokens:
        return [voc[tokens]]
    else:
        return list(itemgetter(*tokens.split(' '))(voc))


def extract():
    for id_model, m in enumerate(args.model_name):
        print('****************************************************************')
        print('EXTRACTION MODEL: %s' % m)

        if args.text_output_split[id_model]:
            # create directories for split
            if not os.path.exists(text_words_features_dir.format(args.dataset, m.lower())):
                os.makedirs(text_words_features_dir.format(args.dataset, m.lower()))

        # model setting
        word2vec_model = gensim.downloader.load(args.model_name[id_model])

        # dataset padding
        data = read_csv(reviews_path.format(args.dataset), sep='\t')

        count_users = data.groupby('USER_ID').size().reset_index(name='counts')
        count_users = count_users.sort_values(by='counts', ascending=False)
        max_reviews_users = count_users.head(1)['counts'].values

        count_items = data.groupby('ITEM_ID').size().reset_index(name='counts')
        count_items = count_items.sort_values(by='counts', ascending=False)
        max_reviews_items = count_items.head(1)['counts'].values

        max_reviews = max([max_reviews_users, max_reviews_items])

        print('Loaded dataset from %s' % reviews_path.format(args.dataset))
        # data['num_tokens'] = data['tokens'].map(lambda x: len(x.split(' ')))
        data['num_tokens'] = data['tokens'].map(lambda x: create_vocabulary(x))
        max_num_tokens = int(data['num_tokens'].max())
        print('Max num of tokens: %d' % max_num_tokens)
        # data['tokens'] = data['tokens'].map(lambda x, max_num=max_num_tokens: pad(x, max_num))
        # print('The dataset has been padded!')

        final_vocabulary = list(set(vocabulary)) + ['<pad>']
        padding_index = len(final_vocabulary) - 1
        final_vocabulary_dict = {k: i for i, k in enumerate(final_vocabulary)}

        print('Starting tokens position calculation...')
        data['tokens_position'] = data['tokens'].map(
            lambda x, voc=final_vocabulary_dict: find_indices_vocabulary(x, voc))
        print('Tokens position calculation has ended!')

        print('Starting to write to tsv file...')
        data = data[['USER_ID', 'ITEM_ID', 'tokens', 'tokens_position', 'num_tokens']]
        write_csv(data, reviews_output_path.format(args.dataset), sep='\t')
        print('Data has been written to tsv file!')

        users_tokens = {}
        items_tokens = {}
        for u in data['USER_ID'].unique().tolist():
            list_of_lists = data[data['USER_ID'] == u]['tokens_position']
            print(list_of_lists)
            exit()
            if args.concat_tokens:
                list_of_tokens = [item for sublist in list_of_lists for item in sublist]
                if len(list_of_tokens) > args.max_tokens:
                    list_of_tokens_padded = list_of_tokens[:args.max_tokens]
                else:
                    list_of_tokens_padded = list_of_tokens + ([padding_index] * (args.max_tokens - len(list_of_tokens)))
            else:
                list_of_tokens_padded = [item[:args.max_tokens] if len(item) > args.max_tokens else item + (
                        [padding_index] * (args.max_tokens - len(item))) for item in list_of_lists]
                if len(list_of_tokens_padded) > max_reviews:
                    list_of_tokens_padded = list_of_tokens_padded[:max_reviews]
                else:
                    list_of_tokens_padded += (
                                [[padding_index] * args.max_tokens] * (max_reviews - len(list_of_tokens_padded)))
            users_tokens[str(u)] = list_of_tokens_padded

        for i in data['ITEM_ID'].unique().tolist():
            list_of_lists = data[data['ITEM_ID'] == i]['tokens_position']
            if args.concat_tokens:
                list_of_tokens = [item for sublist in list_of_lists for item in sublist]
                if len(list_of_tokens) > args.max_tokens:
                    list_of_tokens_padded = list_of_tokens[:args.max_tokens]
                else:
                    list_of_tokens_padded = list_of_tokens + ([padding_index] * (args.max_tokens - len(list_of_tokens)))
            else:
                list_of_tokens_padded = [item[:args.max_tokens] if len(item) > args.max_tokens else item + (
                        [padding_index] * (args.max_tokens - len(item))) for item in list_of_lists]
                if len(list_of_tokens_padded) > max_reviews:
                    list_of_tokens_padded = list_of_tokens_padded[:max_reviews]
                else:
                    list_of_tokens_padded += (
                                [[padding_index] * args.max_tokens] * (max_reviews - len(list_of_tokens_padded)))
            items_tokens[str(i)] = list_of_tokens_padded

        users_filename = 'users_tokens_concat.json' if args.concat_tokens else 'users_tokens_no_concat.json'
        with open('../data/{0}/{1}'.format(args.dataset, users_filename), 'w') as f:
            json.dump(users_tokens, f)

        items_filename = 'items_tokens_concat.json' if args.concat_tokens else 'items_tokens_no_concat.json'
        with open('../data/{0}/{1}'.format(args.dataset, items_filename), 'w') as f:
            json.dump(items_tokens, f)

        len_data = len(data)
        del data, users_tokens, items_tokens

        # text words features
        text_words_features_vocabulary = np.zeros(
            shape=[len(list(final_vocabulary_dict.keys())), word2vec_model.vector_size]
        )

        # features extraction
        print('Starting vocabulary embedding extraction...\n')
        start = time.time()

        for v, idx in final_vocabulary_dict.items():
            try:
                text_words_features_vocabulary[idx] = word2vec_model.get_vector(v, norm=True)
            except KeyError:
                pass

            if (idx + 1) % args.print_each == 0:
                sys.stdout.write('\r%d/%d samples completed' % (idx + 1, len(list(final_vocabulary_dict.keys()))))
                sys.stdout.flush()

        end = time.time()
        print('\n\nFeature extraction completed in %f seconds.' % (end - start))

        if args.normalize:
            text_words_features_vocabulary = text_words_features_vocabulary / np.max(
                np.abs(text_words_features_vocabulary))

        if args.text_output_split[id_model]:
            for d in range(len_data):
                save_np(npy=text_words_features_vocabulary[d],
                        filename=text_words_features_dir.format(args.dataset, m.lower()) + str(d) + '.npy')
            print('Saved text vocabulary words features numpy to ==> %s' %
                  text_words_features_dir.format(args.dataset, m.lower()))
        else:
            save_np(npy=text_words_features_vocabulary,
                    filename=text_words_features_path.format(args.dataset, m.lower()))
            print('Saved text vocabulary words features numpy to ==> %s' %
                  text_words_features_path.format(args.dataset, m.lower()))


if __name__ == '__main__':
    extract()
