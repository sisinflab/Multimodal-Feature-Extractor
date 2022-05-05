import os
import argparse
import re
import json

import itertools
from operator import itemgetter
from collections import Counter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def parse_args():
    parser = argparse.ArgumentParser(description="Run feature extraction for words in text.")
    parser.add_argument('--gpu', type=int, default=0, help='GPU id to run experiments')
    parser.add_argument('--dataset', nargs='?', default='amazon_men', help='dataset path')
    parser.add_argument('--model_name', nargs='+', type=str, default=['word2vec-google-news-300'],
                        help='model for feature extraction')
    parser.add_argument('--normalize', type=bool, default=False, help='whether to normalize output or not')
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


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def pad_sentences(u_text, u_len, u2_len, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_num = u_len
    review_len = u2_len

    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        padded_u_train = []
        for ri in range(review_num):
            if ri < len(u_reviews):
                sentence = u_reviews[ri]
                if review_len > len(sentence):
                    num_padding = review_len - len(sentence)
                    new_sentence = sentence + [padding_word] * num_padding
                    padded_u_train.append(new_sentence)
                else:
                    new_sentence = sentence[:review_len]
                    padded_u_train.append(new_sentence)
            else:
                new_sentence = [padding_word] * review_len
                padded_u_train.append(new_sentence)
        u_text2[i] = padded_u_train

    return u_text2


def pad_reviewid(u_i_dict, u_len, num, priv_pub_map1, priv_pub_map2):
    new_dict = {}
    for k, v in u_i_dict.items():
        while u_len > len(v):
            v.append(num)
        if u_len < len(v):
            v = v[:u_len]
        new_dict[priv_pub_map1[k]] = list(itemgetter(*v)(priv_pub_map2))

    return new_dict


def build_vocab(sentences1, sentences2):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts1 = Counter(itertools.chain(*sentences1))
    # Mapping from index to word
    vocabulary_inv1 = [x[0] for x in word_counts1.most_common()]
    vocabulary_inv1 = list(sorted(vocabulary_inv1))
    # Mapping from word to index
    vocabulary1 = {x: i for i, x in enumerate(vocabulary_inv1)}

    word_counts2 = Counter(itertools.chain(*sentences2))
    # Mapping from index to word
    vocabulary_inv2 = [x[0] for x in word_counts2.most_common()]
    vocabulary_inv2 = list(sorted(vocabulary_inv2))
    # Mapping from word to index
    vocabulary2 = {x: i for i, x in enumerate(vocabulary_inv2)}
    return [vocabulary1, vocabulary_inv1, vocabulary2, vocabulary_inv2]


def build_input_data(u_text, i_text, vocabulary_u, vocabulary_i):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    u_text2 = {}
    for i in u_text.keys():
        u_reviews = u_text[i]
        u = [[vocabulary_u[word] for word in words] for words in u_reviews]
        u_text2[i] = u
    i_text2 = {}
    for j in i_text.keys():
        i_reviews = i_text[j]
        i = [[vocabulary_i[word] for word in words] for words in i_reviews]
        i_text2[j] = i
    return u_text2, i_text2


def extract():
    for id_model, m in enumerate(args.model_name):
        print('****************************************************************')
        print('EXTRACTION MODEL: %s' % m)

        # model setting
        word2vec_model = gensim.downloader.load(args.model_name[id_model])

        # dataset padding
        data = read_csv(reviews_path.format(args.dataset), sep='\t')
        print('Loaded dataset from %s' % reviews_path.format(args.dataset))

        public_users = {u: idx for idx, u in enumerate(sorted(data['USER_ID'].unique().tolist()))}
        public_items = {i: idx for idx, i in enumerate(sorted(data['ITEM_ID'].unique().tolist()))}

        token_users = dict()
        token_items = dict()
        u_rid = {}
        i_rid = {}
        for idx, line in data.iterrows():
            if line['USER_ID'] in token_users:
                pass
            else:
                token_users[line['USER_ID']] = []
                for s in data[data['USER_ID'] == line['USER_ID']]['REVIEW']:
                    s1 = clean_str(s)
                    s1 = s1.split(" ")
                    token_users[line['USER_ID']].append(s1)
                u_rid[line['USER_ID']] = []
                for s in data[data['USER_ID'] == line['USER_ID']]['ITEM_ID']:
                    u_rid[line['USER_ID']].append(s)

            if line['ITEM_ID'] in token_items:
                pass
            else:
                token_items[line['ITEM_ID']] = []
                for s in data[data['ITEM_ID'] == line['ITEM_ID']]['REVIEW']:
                    s1 = clean_str(s)
                    s1 = s1.split(" ")
                    token_items[line['ITEM_ID']].append(s1)
                i_rid[line['ITEM_ID']] = []
                for s in data[data['ITEM_ID'] == line['ITEM_ID']]['USER_ID']:
                    i_rid[line['ITEM_ID']].append(s)

        review_num_u = np.array([len(x) for x in token_users.values()])
        x = np.sort(review_num_u)
        u_len = x[int(0.9 * len(review_num_u)) - 1]
        review_len_u = np.array([len(j) for i in token_users.values() for j in i])
        x2 = np.sort(review_len_u)
        u2_len = x2[int(0.9 * len(review_len_u)) - 1]

        review_num_i = np.array([len(x) for x in token_items.values()])
        y = np.sort(review_num_i)
        i_len = y[int(0.9 * len(review_num_i)) - 1]
        review_len_i = np.array([len(j) for i in token_items.values() for j in i])
        y2 = np.sort(review_len_i)
        i2_len = y2[int(0.9 * len(review_len_i)) - 1]

        user_num = max(data['USER_ID'].unique().tolist())
        item_num = max(data['ITEM_ID'].unique().tolist())
        public_users[user_num + 1] = data['USER_ID'].nunique()
        public_items[item_num + 1] = data['ITEM_ID'].nunique()
        token_users = pad_sentences(token_users, u_len, u2_len)
        u_rid = pad_reviewid(u_rid, u_len, item_num + 1, public_users, public_items)
        token_items = pad_sentences(token_items, i_len, i2_len)
        i_rid = pad_reviewid(i_rid, i_len, user_num + 1, public_items, public_users)

        user_voc = [xx for x in token_users.values() for xx in x]
        item_voc = [xx for x in token_items.values() for xx in x]

        vocabulary_user, vocabulary_inv_user, vocabulary_item, vocabulary_inv_item = build_vocab(user_voc, item_voc)
        token_users, token_items = build_input_data(token_users, token_items, vocabulary_user, vocabulary_item)

        users_filename = 'users_tokens_narre.json'
        with open('../data/{0}/{1}'.format(args.dataset, users_filename), 'w') as f:
            json.dump(token_users, f)

        items_filename = 'items_tokens_narre.json'
        with open('../data/{0}/{1}'.format(args.dataset, items_filename), 'w') as f:
            json.dump(token_items, f)

        users_filename = 'users_pos_narre.json'
        with open('../data/{0}/{1}'.format(args.dataset, users_filename), 'w') as f:
            json.dump(u_rid, f)

        items_filename = 'items_pos_narre.json'
        with open('../data/{0}/{1}'.format(args.dataset, items_filename), 'w') as f:
            json.dump(i_rid, f)

        # text words features
        text_words_features_vocabulary_users = np.zeros(
            shape=[len(list(vocabulary_user.keys())), word2vec_model.vector_size]
        )
        text_words_features_vocabulary_items = np.zeros(
            shape=[len(list(vocabulary_item.keys())), word2vec_model.vector_size]
        )

        # features extraction
        print('Starting vocabulary embedding extraction...\n')
        start = time.time()

        for v, idx in vocabulary_user.items():
            try:
                text_words_features_vocabulary_users[idx] = word2vec_model.get_vector(v, norm=True)
            except KeyError:
                pass

            if (idx + 1) % args.print_each == 0:
                sys.stdout.write(
                    '\r%d/%d samples completed for users' % (idx + 1, len(list(vocabulary_user.keys()))))
                sys.stdout.flush()

        for v, idx in vocabulary_item.items():
            try:
                text_words_features_vocabulary_items[idx] = word2vec_model.get_vector(v, norm=True)
            except KeyError:
                pass

            if (idx + 1) % args.print_each == 0:
                sys.stdout.write(
                    '\r%d/%d samples completed for items' % (idx + 1, len(list(vocabulary_item.keys()))))
                sys.stdout.flush()

        end = time.time()
        print('\n\nFeature extraction completed in %f seconds.' % (end - start))

        if args.normalize:
            text_words_features_vocabulary_users = text_words_features_vocabulary_users / np.max(
                np.abs(text_words_features_vocabulary_users))
            text_words_features_vocabulary_items = text_words_features_vocabulary_items / np.max(
                np.abs(text_words_features_vocabulary_items))

        save_np(npy=text_words_features_vocabulary_users,
                filename='../data/{0}/users_{1}_narre.npy'.format(args.dataset, m.lower()))
        save_np(npy=text_words_features_vocabulary_items,
                filename='../data/{0}/items_{1}_narre.npy'.format(args.dataset, m.lower()))


if __name__ == '__main__':
    extract()
