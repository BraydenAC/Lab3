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
img_X_train = torch.tensor(scaler.fit_transform(pd.read_csv("Datasets/hateful_memes/img_train.csv").values), dtype=torch.float32)
img_X_dev = torch.tensor(scaler.transform(pd.read_csv("Datasets/hateful_memes/img_dev.csv").values), dtype=torch.float32)
img_X_test = torch.tensor(scaler.transform(pd.read_csv("Datasets/hateful_memes/img_test.csv").values), dtype=torch.float32)

text_X_train = torch.tensor(pd.read_csv('Datasets/hateful_memes/text_train.csv').values, dtype=torch.float32)
text_X_dev = torch.tensor(pd.read_csv('Datasets/hateful_memes/text_dev.csv').values, dtype=torch.float32)
text_X_test = torch.tensor(pd.read_csv('Datasets/hateful_memes/text_test.csv').values, dtype=torch.float32)

y_train = torch.tensor((pd.read_json("Datasets/hateful_memes/train.jsonl", lines=True)['label']), dtype=torch.float32).unsqueeze(1)
y_dev = torch.tensor((pd.read_json("Datasets/hateful_memes/dev.jsonl", lines=True))['label'], dtype=torch.float32)
y_test = torch.tensor((pd.read_json("Datasets/hateful_memes/test_seen.jsonl", lines=True)['label']), dtype=torch.float32)

#initialize model
text_input_dim = 768
img_input_dim = 2048
hidden_dim = 512
output_dim = 1

img_model = TextClassifier(img_input_dim, hidden_dim, output_dim)
text_model = TextClassifier(text_input_dim, hidden_dim, output_dim)

#Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

#Adam optimizer
text_optimizer = torch.optim.Adam(text_model.parameters(), lr=1e-3)
img_optimizer = torch.optim.Adam(img_model.parameters(), lr=1e-3)

#Do Training Loop
num_epochs = 5
threshold = 0.4

#Split data into mini-batches
print(img_X_train.shape)
print(text_X_train.shape)
print(y_train.shape)
batchable_train_data = TensorDataset(img_X_train, text_X_train, y_train)
train_dataloader = DataLoader(batchable_train_data, batch_size=64, shuffle=True)

#Training Loop
for epoch in range(num_epochs):
    # set to training mode
    img_model.train()
    text_model.train()
    added_loss = 0.0

    #process
    for img_X_batch, text_X_batch, y_batch in train_dataloader:
        #zero out gradients
        img_optimizer.zero_grad()
        text_optimizer.zero_grad()

        #get logits/outputs
        img_outputs = img_model(img_X_batch)
        text_outputs = text_model(text_X_batch)

        #combine outputs
        combined_outputs = (img_outputs + text_outputs)/2

        #compute loss
        loss = criterion(combined_outputs, y_batch)
        loss.backward()

        #add loss to counter
        added_loss += loss.item()

        img_optimizer.step()
        text_optimizer.step()


    average_loss = added_loss/len(train_dataloader)

    #print progress
    img_model.eval()
    with torch.no_grad():
        print(f"Epoch {epoch + 1}/{num_epochs}, Combined Loss: {average_loss}")




#Evaluate on dev and test sets
img_model.eval()
text_model.eval()
with torch.no_grad():
    #Image Probabilities
    img_dev_outputs = img_model(img_X_dev)
    img_dev_probabilities = torch.sigmoid(img_dev_outputs)

    img_test_outputs = img_model(img_X_test)
    img_test_probabilities = torch.sigmoid(img_test_outputs)

    #Text Probabilities
    text_dev_outputs = text_model(text_X_dev)
    text_dev_probabilities = torch.sigmoid(text_dev_outputs)

    text_test_outputs = text_model(text_X_test)
    text_test_probabilities = torch.sigmoid(text_test_outputs)

    #Combined Probabilities and Predictions
    dev_probabilities = (img_dev_probabilities + text_dev_probabilities)/2
    dev_predictions = (dev_probabilities >= threshold).float()
    test_probabilities = (img_test_probabilities + text_test_probabilities)/2
    test_predictions = (test_probabilities >= threshold).float()

    #Calculate metrics
    # dev_accuracy = accuracy_score(y_dev, dev_predictions)
    # dev_precision = precision_score(y_dev, dev_predictions)
    # dev_recall = recall_score(y_dev, dev_predictions)
    # dev_f1 = f1_score(y_dev, dev_predictions)
    # dev_auc_roc = roc_auc_score(y_dev, dev_probabilities)
    #
    # print("dev results")
    # print(f"Accuracy:  {dev_accuracy}")
    # print(f"Precision: {dev_precision}")
    # print(f"Recall:    {dev_recall}")
    # print(f"F1 Score:  {dev_f1}")
    # print(f"AUC_ROC:   {dev_auc_roc}")

    test_accuracy = accuracy_score(y_test, test_predictions)
    test_precision = precision_score(y_test, test_predictions)
    test_recall = recall_score(y_test, test_predictions)
    test_f1 = f1_score(y_test, test_predictions)
    test_auc_roc = roc_auc_score(y_test, test_probabilities)

    print("test results")
    print(f"Accuracy:  {test_accuracy}")
    print(f"Precision: {test_precision}")
    print(f"Recall:    {test_recall}")
    print(f"F1 Score:  {test_f1}")
    print(f"AUC_ROC:   {test_auc_roc}")

