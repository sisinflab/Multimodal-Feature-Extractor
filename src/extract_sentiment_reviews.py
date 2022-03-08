from transformers import pipeline
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

dataset = 'baby'
train_reviews = pd.read_csv(f'../data/{dataset}/train_reviews.txt', sep='\t', header=None)
sentiment_pipeline = pipeline(task="sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
model = list(sentiment_pipeline.model.children())[-3]
model.eval()
tokenizer = sentiment_pipeline.tokenizer

if not os.path.exists(f'../data/{dataset}/reviews/'):
    os.makedirs(f'../data/{dataset}/reviews/')
with tqdm(total=len(train_reviews)) as t:
    for idx, row in train_reviews.iterrows():
        inputs = tokenizer.encode_plus(row[2], add_special_tokens=True, truncation=True, return_tensors="pt")
        np.save(f'../data/{dataset}/reviews/{row[1]}_{row[0]}.npy',
                model(**inputs).pooler_output.detach().cpu().numpy())
        t.update()
