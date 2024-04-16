import torch
import torch.nn as nn
import torch.optim as optim

from extempconvsm import ExTempConvSM
from dataset import scaffold_loaders
from constants import DEVICE
from constants import NUM_EPOCHS

# We use ExTempConvSM. This can be easily replaced with ExTempConvLG
model = ExTempConvSM()
model = model.to(DEVICE)

# Define data
train_data = None
test_data = None
train_labels = None
test_labels = None

train_loader, test_loader = scaffold_loaders(
    train_data, train_labels, test_data, test_labels
)

loss_fn = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters())

history = {"loss": [], "test_loss": [], "acc": [], "test_acc": []}

for epoch in range(NUM_EPOCHS):
    correct, total = 0, 0
    running_loss = 0.0

    model.train()
    for x, y in train_loader:
        yhat = model(x)

        opt.zero_grad()
        loss = loss_fn(yhat, y)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        _, predictions = torch.max(yhat, 1)
        total += x.size(0)
        correct += (predictions == y).sum().item()

    running_loss /= len(train_loader)
    acc = 100.0 * correct / total

    history["loss"].append(running_loss)
    history["acc"].append(acc)

    print(f"Epoch: {epoch+1} | Training Loss: {running_loss} Accuracy: {acc}", end=" ")

    running_loss = 0.0
    correct, total = 0, 0

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            yhat = model(x)
            loss = loss_fn(yhat, y)
            _, predicted = torch.max(yhat, 1)

            running_loss += loss.item()
            total += x.size(0)
            correct += (predictions == y).sum().item()

        running_loss /= len(test_loader)
        acc = 100.0 * correct / total

        history["test_loss"].append(running_loss)
        history["test_acc"].append(acc)

        print(f"Testing Loss: {running_loss} Accuracy: {acc}")
