# QSAR with PyTorch and RDKit (to be pasted into a Jupyter Notebook)

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv("data/sample_molecules.csv")

# Convert SMILES to Morgan fingerprints
def smiles_to_fp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    else:
        return None

fps = []
labels = []
for i, row in df.iterrows():
    fp = smiles_to_fp(row['smiles'])
    if fp:
        arr = torch.tensor(list(fp), dtype=torch.float32)
        fps.append(arr)
        labels.append(torch.tensor(row['label'], dtype=torch.float32))

X = torch.stack(fps)
y = torch.stack(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch Dataset
class MolDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(MolDataset(X_train, y_train), batch_size=4, shuffle=True)
test_loader = DataLoader(MolDataset(X_test, y_test), batch_size=4)

# Neural network
class QSARModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 1)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.out(x)).squeeze()

model = QSARModel(2048)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# Training loop
for epoch in range(10):
    model.train()
    for xb, yb in train_loader:
        preds = model(xb)
        loss = loss_fn(preds, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
with torch.no_grad():
    preds = []
    actual = []
    for xb, yb in test_loader:
        out = model(xb)
        preds += list((out > 0.5).float())
        actual += list(yb)
    acc = accuracy_score(actual, preds)
    print(f"Test Accuracy: {acc:.2f}")
