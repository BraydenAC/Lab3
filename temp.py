import pandas as pd

seen = pd.read_json("Datasets/hateful_memes/test_seen.jsonl", lines=True)
normal = pd.read_json("Datasets/hateful_memes/test.jsonl", lines=True)

new_normal = pd.concat([normal, seen['label']], axis=1)

new_normal.to_json("Datasets/hateful_memes/new_test.jsonl")



X_train = torch.tensor(pd.concat([pd.read_csv('Datasets/hateful_memes/text_train.csv'),
                                       pd.read_csv('Datasets/hateful_memes/img_train.csv')], axis=1).values, dtype=torch.float32)
X_dev = torch.tensor(pd.concat([pd.read_csv('Datasets/hateful_memes/text_dev.csv'),
                                       pd.read_csv('Datasets/hateful_memes/img_dev.csv')], axis=1).values, dtype=torch.float32)
X_test = torch.tensor(pd.concat([pd.read_csv('Datasets/hateful_memes/text_test.csv'),
                                       pd.read_csv('Datasets/hateful_memes/img_test.csv')], axis=1).values, dtype=torch.float32)
y_train = torch.tensor((pd.read_json("Datasets/hateful_memes/train.jsonl", lines=True)['label']), dtype=torch.float32).unsqueeze(1)
y_dev = torch.tensor((pd.read_json("Datasets/hateful_memes/dev.jsonl", lines=True))['label'], dtype=torch.float32)
y_test = torch.tensor((pd.read_json("Datasets/hateful_memes/test_seen.jsonl", lines=True)['label']), dtype=torch.float32)