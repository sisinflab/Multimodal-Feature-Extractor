import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

dataset = 'baby'
device = "cuda:0" if torch.cuda.is_available() else "cpu"
train_reviews = pd.read_csv(f'../data/{dataset}/train_reviews.txt', sep='\t', header=None)
sentiment_pipeline = pipeline(task="sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
model = list(sentiment_pipeline.model.children())[-3]
model.eval()
model.to(device)
tokenizer = sentiment_pipeline.tokenizer.to(device)

if not os.path.exists(f'../data/{dataset}/reviews/'):
    os.makedirs(f'../data/{dataset}/reviews/')
with tqdm(total=len(train_reviews)) as t:
    for idx, row in train_reviews.iterrows():
        inputs = tokenizer.encode_plus(row[2], truncation=True, return_tensors="pt").to(device)
        np.save(f'../data/{dataset}/reviews/{row[1]}_{row[0]}.npy',
                model(**inputs.to(device)).pooler_output.detach().cpu().numpy())
        t.update()
