import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[*] Device:", device)

# Load data
X_train = np.load("X_train.npy")
X_val   = np.load("X_val.npy")
X_test  = np.load("X_test.npy")

y_train = np.load("y_train.npy")
y_val   = np.load("y_val.npy")
y_test  = np.load("y_test.npy")

seq_len = X_train.shape[1]
num_features = X_train.shape[2]

# DataLoaders
train_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    ),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    ),
    batch_size=64
)

test_loader = DataLoader(
    TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    ),
    batch_size=64
)

# LSTM Classifier Model
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return self.sigmoid(out).squeeze()

model = LSTMClassifier(num_features).to(device)

# Training setup
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10
for epoch in range(1, epochs + 1):
    model.train()
    losses = []

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch} | Loss: {np.mean(losses):.4f}")

# Evaluation function
def evaluate(loader, y_true, name):
    model.eval()
    preds_all = []

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            preds_all.extend((preds > 0.5).cpu().numpy())

    acc = accuracy_score(y_true, preds_all)
    f1  = f1_score(y_true, preds_all)
    cm  = confusion_matrix(y_true, preds_all)

    print(f"\n--- {name} ---")
    print("Accuracy:", acc)
    print("F1:", f1)
    print(cm)

# Results
evaluate(val_loader, y_val, "VAL")
evaluate(test_loader, y_test, "TEST")
