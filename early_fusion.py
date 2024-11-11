import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

#Define a simple fully connected neural network
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fci = nn.Linear(input_dim, hidden_dim)

        #Activation function between layers
        self.relu = nn.ReLU()
        self.fcl = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fci(x)
        x = self.relu(x)
        x = self.fcl(x)
        return x
X_train = torch.tensor(pd.concat([pd.read_csv('Datasets/hateful_memes/text_train.csv'),
                                       pd.read_csv('Datasets/hateful_memes/img_train.csv')], axis=1).values, dtype=torch.float32)
X_dev = torch.tensor(pd.concat([pd.read_csv('Datasets/hateful_memes/text_dev.csv'),
                                       pd.read_csv('Datasets/hateful_memes/img_dev.csv')], axis=1).values, dtype=torch.float32)
X_test = torch.tensor(pd.concat([pd.read_csv('Datasets/hateful_memes/text_test.csv'),
                                       pd.read_csv('Datasets/hateful_memes/img_test.csv')], axis=1).values, dtype=torch.float32)
y_train = torch.tensor((pd.read_json("Datasets/hateful_memes/train.jsonl", lines=True)['label']), dtype=torch.float32).unsqueeze(1)
y_dev = torch.tensor((pd.read_json("Datasets/hateful_memes/dev.jsonl", lines=True))['label'], dtype=torch.float32)
y_test = torch.tensor((pd.read_json("Datasets/hateful_memes/test_seen.jsonl", lines=True)['label']), dtype=torch.float32)

#initialize model
input_dim = 2816
hidden_dim = 512
output_dim = 1

model = TextClassifier(input_dim, hidden_dim, output_dim)

#Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

#Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Do Training Loop
num_epochs = 8
train_file = pd.read_json('Datasets/hateful_memes/train.jsonl', lines=True)
y_train = torch.tensor(train_file['label'], dtype=torch.float32).unsqueeze(1)

#Training Loop
for epoch in range(num_epochs):
    # set to training mode
    model.train()

    #process
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    #print progress
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


#Evaluate on dev and test sets
model.eval()
threshold = 0.3
with torch.no_grad():
    dev_outputs = model(X_dev)
    dev_file = pd.read_json('Datasets/hateful_memes/dev.jsonl', lines=True)
    y_dev = torch.tensor(dev_file['label'], dtype=torch.float32)

    dev_probabilities = torch.sigmoid(dev_outputs)
    dev_predictions = (dev_probabilities >= threshold).float()

    test_outputs = model(X_test)
    test_file = pd.read_json('Datasets/hateful_memes/test_seen.jsonl', lines=True)
    y_test = torch.tensor(test_file['label'], dtype=torch.float32)

    test_probabilities = torch.sigmoid(test_outputs)
    test_predictions = (test_probabilities >= threshold).float()

    #Calculate metrics
    # dev_accuracy = accuracy_score(y_dev, dev_predictions)
    # dev_precision = precision_score(y_dev, dev_predictions)
    # dev_recall = recall_score(y_dev, dev_predictions)
    # dev_f1 = f1_score(y_dev, dev_predictions)
    # dev_auc_roc = roc_auc_score(y_dev, dev_probabilities)
    #
    # print("dev results")
    # print(f"Accuracy: {dev_accuracy}")
    # print(f"Precision: {dev_precision}")
    # print(f"Recall: {dev_recall}")
    # print(f"F1 Score: {dev_f1}")
    # print(f"AUC_ROC: {dev_auc_roc}")

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    test_auc_roc = roc_auc_score(y_test, torch.sigmoid(test_outputs))

    print("test results")
    print(f"Accuracy: {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1 Score: {test_f1}")
    print(f"AUC_ROC: {test_auc_roc}")

