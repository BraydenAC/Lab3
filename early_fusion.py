import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

#Define a simple fully connected neural network
class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        self.fcl = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fcl(x)
        return x

scaler = StandardScaler()
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
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

#Do Training Loop
num_epochs = 40
threshold = 0.3

#Split data into mini-batches
batchable_train_data = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(batchable_train_data, batch_size=64, shuffle=True)

#Training Loop
for epoch in range(num_epochs):
    # set to training mode
    model.train()
    added_loss = 0.0

    #process
    for X_batch, y_batch in train_dataloader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        added_loss += loss.item()
        loss.backward()
        optimizer.step()

    average_loss = added_loss/len(train_dataloader)

    #print progress
    model.eval()
    with torch.no_grad():
        # train_probabilities = torch.sigmoid(accumulated_outputs)
        # train_predictions = (train_probabilities >= threshold).float()
        print(f"Epoch {epoch}/{num_epochs}, Loss: {average_loss}")


#Evaluate on dev and test sets
model.eval()
with torch.no_grad():
    dev_outputs = model(X_dev)
    dev_probabilities = torch.sigmoid(dev_outputs)
    dev_predictions = (dev_probabilities >= threshold).float()
    print(dev_probabilities[0], dev_probabilities[1], dev_probabilities[2], dev_probabilities[3], dev_probabilities[4])
    print(dev_predictions[0], dev_predictions[1], dev_predictions[2], dev_predictions[3], dev_predictions[4])

    test_outputs = model(X_test)
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
    test_auc_roc = roc_auc_score(y_test, test_probabilities)

    print("test results")
    print(f"Accuracy: {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall: {test_recall}")
    print(f"F1 Score: {test_f1}")
    print(f"AUC_ROC: {test_auc_roc}")

