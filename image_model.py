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
training_set = torch.tensor(pd.read_csv('Datasets/hateful_memes/text_train.csv').values, dtype=torch.float32)
dev_set = torch.tensor(pd.read_csv('Datasets/hateful_memes/text_dev.csv').values, dtype=torch.float32)
test_set = torch.tensor(pd.read_csv('Datasets/hateful_memes/text_test.csv').values, dtype=torch.float32)

#initialize model
input_dim = 2048
hidden_dim = 512
output_dim = 1

model = TextClassifier(input_dim, hidden_dim, output_dim)

#Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

#Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Do Training Loop
num_epochs = 20
train_file = pd.read_json('Datasets/hateful_memes/train.jsonl', lines=True)
train_labels = torch.tensor(train_file['label'], dtype=torch.float32).unsqueeze(1)

#Training Loop
for epoch in range(num_epochs):
    # set to training mode
    model.train()

    #process
    optimizer.zero_grad()
    outputs = model(training_set)
    loss = criterion(outputs, train_labels)
    loss.backward()
    optimizer.step()

    #print progress
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")


#Evaluate on dev and test sets
model.eval()
with torch.no_grad():
    dev_outputs = model(dev_set)
    dev_file = pd.read_json('Datasets/hateful_memes/dev_seen.jsonl', lines=True)
    dev_labels = torch.tensor(dev_file['label'], dtype=torch.float32)
    dev_predictions = torch.sigmoid(dev_outputs).round()

    test_outputs = model(test_set)
    test_file = pd.read_json('Datasets/hateful_memes/test_seen.jsonl', lines=True)
    test_labels = torch.tensor(test_file['label'], dtype=torch.float32)
    test_predictions = torch.sigmoid(test_outputs).round()

    #Calculate metrics
    dev_accuracy = accuracy_score(dev_labels, dev_predictions)
    dev_precision = precision_score(dev_labels, dev_predictions)
    dev_recall = recall_score(dev_labels, dev_predictions)
    dev_f1 = f1_score(dev_labels, dev_predictions)
    dev_auc_roc = roc_auc_score(dev_labels, torch.sigmoid(dev_outputs))

    print("dev results")
    print(f"Accuracy: {dev_accuracy}")
    print(f"Precision: {dev_precision}")
    print(f"Recall: {dev_recall}")
    print(f"F1 Score: {dev_f1}")
    print(f"AUC_ROC: {dev_auc_roc}")

    # test_accuracy = accuracy_score(test_labels, test_predictions)
    # test_precision = precision_score(test_labels, test_predictions)
    # test_recall = recall_score(test_labels, test_predictions)
    # test_f1 = f1_score(test_labels, test_predictions)
    # test_auc_roc = roc_auc_score(test_labels, torch.sigmoid(test_outputs))
    #
    # print("test results")
    # print(f"Accuracy: {test_accuracy}")
    # print(f"Precision: {test_precision}")
    # print(f"Recall: {test_recall}")
    # print(f"F1 Score: {test_f1}")
    # print(f"AUC_ROC: {test_auc_roc}")

