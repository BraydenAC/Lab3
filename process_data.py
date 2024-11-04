import torch
from transformers import AutoTokenizer , AutoModel
import pandas as pd
import json
import numpy as np


# Tokenize text
#Initialize pretrained model
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
headers = ['id', 'img', 'label', 'text']

model.eval()

# def tokenize_text(texts):
#     #enact tokenization
#     inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
#
#     #Run through BERT
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     #extract sentence embeddings
#     sentence_embeddings = outputs.last_hidden_state[:, 0, :]
#
#     print(sentence_embeddings.shape)
#     return sentence_embeddings

#Replacement function written by chatGPT because the first one made my IDE crash
def tokenize_text_in_batches(texts, batch_size=50):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(batch_embeddings)
    return np.vstack(embeddings)


#Load in data
print("Loading data...")
unprocessed_train = pd.read_json('Datasets/hateful_memes/train.jsonl', lines=True)
unprocessed_dev = pd.read_json('Datasets/hateful_memes/dev_seen.jsonl', lines=True)
unprocessed_test = pd.read_json('Datasets/hateful_memes/test_seen.jsonl', lines=True)

#process data
print("processing training data...")
processed_train = pd.DataFrame(tokenize_text_in_batches(unprocessed_train['text'].tolist()))
print("processing dev data...")
processed_dev = pd.DataFrame(tokenize_text_in_batches(unprocessed_dev['text'].tolist()))
print("processing test data...")
processed_test = pd.DataFrame(tokenize_text_in_batches(unprocessed_test['text'].tolist()))

#save to processed files
print("Saving data...")
processed_train.to_csv('Datasets/hateful_memes/train.csv', index=False)
processed_dev.to_csv('Datasets/hateful_memes/dev.csv', index=False)
processed_test.to_csv('Datasets/hateful_memes/test.csv', index=False)
print("Done!")