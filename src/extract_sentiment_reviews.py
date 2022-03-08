import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Run sentiment feature extraction from reviews.")
parser.add_argument('--dataset', nargs='?', default='baby', help='dataset path')
args = parser.parse_args()
dataset = args.dataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_reviews = pd.read_csv(f'../data/{dataset}/train_reviews.txt', sep='\t', header=None)
train_reviews.columns = ['user', 'item', 'review']
sentiment_pipeline = pipeline(task="sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
users_items_nan = train_reviews[train_reviews['review'].isna()][['user', 'item']]
model = list(sentiment_pipeline.model.children())[-3]
model.eval()
model.to(device)
tokenizer = sentiment_pipeline.tokenizer

if not os.path.exists(f'../data/{dataset}/reviews/'):
    os.makedirs(f'../data/{dataset}/reviews/')
with tqdm(total=len(train_reviews)) as t:
    for idx, row in train_reviews.iterrows():
        if len(users_items_nan[
                   (users_items_nan['user'] == row['user']) & (users_items_nan['item'] == row['item'])]) > 0:
            first_review = train_reviews[train_reviews['user'] == row['user']].dropna().iloc[0, 2]
            inputs = tokenizer.encode_plus(first_review, truncation=True, return_tensors="pt").to(device)
        else:
            inputs = tokenizer.encode_plus(row['review'], truncation=True, return_tensors="pt").to(device)
        np.save(f'../data/{dataset}/reviews/{row["item"]}_{row["user"]}.npy',
                model(**inputs.to(device)).pooler_output.detach().cpu().numpy())
        t.update()
