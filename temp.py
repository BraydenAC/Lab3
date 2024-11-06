import pandas as pd
print("Json file:")
print(pd.read_json('Datasets/hateful_memes/dev_seen.jsonl',lines=True).shape)

print("Json file:")
print(pd.read_json('Datasets/hateful_memes/train.jsonl',lines=True).shape)

print("")
print("CSV file:")
print(pd.read_csv('Datasets/hateful_memes/text_dev.csv').shape)

print("")
print("CSV file:")
print(pd.read_csv('Datasets/hateful_memes/text_train.csv').shape)