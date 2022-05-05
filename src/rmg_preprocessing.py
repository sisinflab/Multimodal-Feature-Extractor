import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
import random
import gensim.downloader
import sys

random.seed(1234)


def dataframe_to_dict(data):
    ratings = data.set_index('USER_ID')[['ITEM_ID', 'RATING']].apply(lambda x: (x['ITEM_ID'], float(x['RATING'])), 1) \
        .groupby(level=0).agg(lambda x: dict(x.values)).to_dict()
    return ratings


dataset = 'amazon_baby'
rawdata = pd.read_csv('../data/{0}/train_reviews.tsv'.format(dataset), sep='\t').drop(columns=['TOKENS']).to_dict(
    'records')

train_dict = dataframe_to_dict(pd.read_csv('../data/{0}/train_reviews.tsv'.format(dataset), sep='\t'))
users = list(train_dict.keys())
items = list({int(k) for a in train_dict.values() for k in a.keys()})
private_users = {p: u for p, u in enumerate(users)}
public_users = {v: k for k, v in private_users.items()}
private_items = {p: i for p, i in enumerate(items)}
public_items = {v: k for k, v in private_items.items()}

word2vec_model = gensim.downloader.load('word2vec-google-news-300')

for i in range(len(rawdata)):
    rawdata[i]['text'] = [word_tokenize(x) for x in sent_tokenize(rawdata[i]['REVIEW'].lower())]

word_dict = {'PADDING': [0, 999999]}
for i in rawdata:
    for k in i['text']:
        for j in k:
            if j in word_dict:
                word_dict[j][1] += 1
            else:
                word_dict[j] = [len(word_dict), 1]

word_dict_freq = {}
for x in word_dict:
    if word_dict[x][1] >= 10:
        word_dict_freq[x] = [len(word_dict_freq), word_dict[x][1]]
print(len(word_dict_freq), len(word_dict))

embed_vocabulary = np.zeros(shape=[len(list(word_dict_freq.keys())), 300])

cnt = 0
for key, _ in word_dict_freq.items():
    if key == 'PADDING':
        pass
    else:
        try:
            embed_vocabulary[cnt] = word2vec_model.get_vector(key, norm=True)
        except KeyError:
            embed_vocabulary[cnt] = np.zeros(300, dtype=np.int)

    if (cnt + 1) % 100 == 0:
        sys.stdout.write(
            '\r%d/%d samples completed' % (cnt + 1, len(list(word_dict_freq.keys()))))
        sys.stdout.flush()
    cnt += 1

mu = np.mean(embed_vocabulary, axis=0)
Sigma = np.cov(embed_vocabulary.T)

norm = np.random.multivariate_normal(mu, Sigma, 1)

for i in range(embed_vocabulary.shape[0]):
    if type(embed_vocabulary[i]) == int:
        embed_vocabulary[i] = np.reshape(norm, 300)
embed_vocabulary[0] = np.zeros(300, dtype='float32')
embed_vocabulary = np.array(embed_vocabulary, dtype='float32')
np.save('../data/{0}/embed_vocabulary.npy'.format(dataset), embed_vocabulary)

MAX_SENT_LENGTH = -np.inf
MAX_SENTS = -np.inf
uir_triples = []
for i in rawdata:
    temp = {}
    doc = []
    for y in i['text']:
        current_doc = [word_dict_freq[x][0] for x in y if x in word_dict_freq]
        doc.append(current_doc)
        if len(current_doc) > MAX_SENT_LENGTH:
            MAX_SENT_LENGTH = len(current_doc)
    if len(doc) > MAX_SENTS:
        MAX_SENTS = len(doc)
    temp['text'] = doc
    temp['item'] = public_items[i['ITEM_ID']]
    temp['user'] = public_users[i['USER_ID']]
    temp['label'] = i['RATING']
    uir_triples.append(temp)

for i in range(len(uir_triples)):
    uir_triples[i]['id'] = i

MAX_SENT_LENGTH = 10 if MAX_SENT_LENGTH > 10 else MAX_SENT_LENGTH
MAX_SENTS = 5 if MAX_SENTS > 5 else MAX_SENTS

item_review_id = {}
user_review_id = {}
item_review_number = {}
user_review_number = {}

for i in uir_triples:
    if i['item'] in item_review_id:
        item_review_id[i['item']].append(i['id'])
        item_review_number[i['item']] = item_review_number[i['item']] + 1
    else:
        item_review_id[i['item']] = [i['id']]
        item_review_number[i['item']] = 1

    if i['user'] in user_review_id:
        user_review_id[i['user']].append(i['id'])
        user_review_number[i['user']] = user_review_number[i['user']] + 1
    else:
        user_review_id[i['user']] = [i['id']]
        user_review_number[i['user']] = 1

MAX_REVIEW_USER = 15 if max(list(user_review_number.values())) > 15 else max(list(user_review_number.values()))
MAX_REVIEW_ITEM = 20 if max(list(item_review_number.values())) > 20 else max(list(item_review_number.values()))
all_user_texts = []
for i in user_review_id:
    pad_docs = []
    for j in user_review_id[i][:MAX_REVIEW_USER]:
        sents = [x[:MAX_SENT_LENGTH] for x in uir_triples[j]['text'][:MAX_SENTS]]
        pad_sents = [x + (MAX_SENT_LENGTH - len(x)) * [0] for x in sents]
        pad_docs.append(pad_sents + [[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(pad_sents)))
    all_user_texts.append(pad_docs + [[[0] * MAX_SENT_LENGTH] * MAX_SENTS] * (MAX_REVIEW_USER - len(pad_docs)))

all_item_texts = []
for i in item_review_id:
    pad_docs = []
    for j in item_review_id[i][:MAX_REVIEW_ITEM]:
        sents = [x[:MAX_SENT_LENGTH] for x in uir_triples[j]['text'][:MAX_SENTS]]
        pad_sents = [x + (MAX_SENT_LENGTH - len(x)) * [0] for x in sents]
        pad_docs.append(pad_sents + [[0] * MAX_SENT_LENGTH] * (MAX_SENTS - len(pad_sents)))
    all_item_texts.append(pad_docs + [[[0] * MAX_SENT_LENGTH] * MAX_SENTS] * (MAX_REVIEW_ITEM - len(pad_docs)))

all_user_texts = np.array(all_user_texts, dtype='int32')
all_item_texts = np.array(all_item_texts, dtype='int32')
np.save('../data/{0}/all_user_texts.npy'.format(dataset), all_user_texts)
np.save('../data/{0}/all_item_texts.npy'.format(dataset), all_item_texts)

item_to_user_id = {}

for i in uir_triples:
    if i['item'] in item_to_user_id:
        item_to_user_id[i['item']].append(i['user'])
    else:
        item_to_user_id[i['item']] = [i['user']]

user_to_item_id = {}

for i in uir_triples:
    if i['user'] in user_to_item_id:
        user_to_item_id[i['user']].append(i['item'])
    else:
        user_to_item_id[i['user']] = [i['item']]

MAX_NEIGHBOR = 20
user_to_item_to_user = []
user_to_item = []
for i in user_to_item_id:
    ids = []

    ui_ids = user_to_item_id[i][:MAX_NEIGHBOR]

    for j in user_to_item_id[i]:
        randids = random.sample(item_to_user_id[j], min(MAX_NEIGHBOR, len(item_to_user_id[j])))

        ids.append(randids + [len(user_to_item_id) - 1] * (MAX_NEIGHBOR - len(randids)))
    ids = ids[:MAX_NEIGHBOR]
    user_to_item_to_user.append(ids + [[len(user_to_item_id) - 1] * MAX_NEIGHBOR] * (MAX_NEIGHBOR - len(ids)))

    user_to_item.append(ui_ids + [len(item_to_user_id) - 1] * (MAX_NEIGHBOR - len(ui_ids)))

item_to_user_to_item = []
item_to_user = []
for i in item_to_user_id:
    ids = []

    iu_ids = item_to_user_id[i][:MAX_NEIGHBOR]

    for j in item_to_user_id[i]:
        randids = random.sample(user_to_item_id[j], min(MAX_NEIGHBOR, len(user_to_item_id[j])))

        ids.append(randids + [len(item_to_user_id) - 1] * (MAX_NEIGHBOR - len(randids)))
    ids = ids[:MAX_NEIGHBOR]
    item_to_user_to_item.append(ids + [[len(item_to_user_id) - 1] * MAX_NEIGHBOR] * (MAX_NEIGHBOR - len(ids)))

    item_to_user.append(iu_ids + [len(user_to_item_id) - 1] * (MAX_NEIGHBOR - len(iu_ids)))

user_to_item_to_user = np.array(user_to_item_to_user, dtype='int32')
user_to_item = np.array(user_to_item, dtype='int32')
item_to_user_to_item = np.array(item_to_user_to_item, dtype='int32')
item_to_user = np.array(item_to_user, dtype='int32')

np.save('../data/{0}/user_to_item_to_user.npy'.format(dataset), user_to_item_to_user)
np.save('../data/{0}/item_to_user_to_item.npy'.format(dataset), item_to_user_to_item)
np.save('../data/{0}/user_to_item.npy'.format(dataset), user_to_item)
np.save('../data/{0}/item_to_user.npy'.format(dataset), item_to_user)
