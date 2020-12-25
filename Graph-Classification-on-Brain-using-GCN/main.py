"""
Code Reference:

PyTorch Geometric Team
Graph Classification Colab Notebook (PyTorch Geometric)
https://colab.research.google.com/drive/1I8a0DfQ3fI7Njc62__mVXUlcAleUclnb?usp=sharing 

"""

import torch
from utils import get_data
from utils import create_dataset
from torch_geometric.data import DataLoader
from model import GCN

print("\n---------Starting to load Data---------\n")

train_data = get_data("Training")
test_data = get_data("Testing")

training_dataset = create_dataset(train_data)
testing_dataset = create_dataset(test_data)

print("\n-----------Data loaded-----------\n")

train_loader = DataLoader(training_dataset, batch_size=5, shuffle = True)
test_loader = DataLoader(testing_dataset, batch_size=5, shuffle=True)

model = GCN(hidden_channels=64)
print("Model:\n\t",model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

def train(model_to_train, train_dataset_loader, loss_function, model_optimizer):
    model_to_train.train()

    for data in train_dataset_loader:  # Iterate in batches over the training dataset.
        out = model_to_train(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = loss_function(out, data.y)  # Computing the loss.
        loss.backward()  # Deriving gradients.
        model_optimizer.step()  # Updating parameters based on gradients.
        model_optimizer.zero_grad()  # Clearing gradients.

def test(model, loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 51):
    train(model, train_loader, criterion, optimizer)
    train_acc = test(model, train_loader)
    test_acc = test(model, test_loader)
    if epoch % 10 == 0:
#         print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}')
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
