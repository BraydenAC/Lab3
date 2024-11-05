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
        x = self.fc2(x)
        return x
training_set = pd.read_csv('Datasets/hateful_memes/text_train.csv')
dev_set = pd.read_csv('Datasets/hateful_memes/text_dev.csv')

#initialize model
input_dim = 768
hidden_dim = 512
output_dim = 1

model = TextClassifier(input_dim, hidden_dim, output_dim)

#Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()

#Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#Do Training Loop
#num_epochs = TODO (defines the number of passes through dataset)
#labels = torch.tensor([0, 1], dtype=torch.float32).unsqueeze(1) <--something needs adjustment here

# set to training mode
model.train()
for epoch in range(numEpochs):
    optimizer.zero_grad()
    outputs = model(sentence_embeddings)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    #print progress
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}") #TODO: add batch number to tracking

#Evaluate on dev set
model.eval()
with torch.no_grad():
    dev_outputs = model(dev_set) #TODO: Fix this

    dev_predictions = torch.sigmoid(dev_outputs).round()

    #Calculate metrics
