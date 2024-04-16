import torch
import torch.nn as nn
import torch.optim as optim

from extempauto import ExTempAuto
from dataset import scaffold_loaders
from constants import DEVICE
from constants import NUM_EPOCHS

model = ExTempAuto()
model = model.to(DEVICE)

# Define data
train_data = None
test_data = None
train_labels = None
test_labels = None

train_loader, test_loader = scaffold_loaders(
    train_data, train_labels, test_data, test_labels
)

c_loss = nn.CrossEntropyLoss()
a_loss = nn.MSELoss()
opt = optim.Adam(model.parameters())

history = {"loss": [], "test_loss": [], "acc": [], "test_acc": []}

for epoch in range(NUM_EPOCHS):
    print(f"Epoch: {epoch}", end=" ")
    correct, total = 0, 0
    running_loss = 0.0

    for x, y in train_loader:
        softmax, decoded = model(x)

        _, predictions = torch.max(softmax, 1)
        total += x.size(0)
        correct += (predictions == y).sum().item()

        opt.zero_grad()
        c = c_loss(softmax, y)
        a = a_loss(decoded, x)
        t = a + c
        t.backward()
        opt.step()
        running_loss += t.item()

    running_loss /= len(train_loader)
    acc = 100.0 * correct / total
    history["loss"].append(running_loss)
    history["acc"].append(acc)

    print(f"Training Loss: {running_loss} Accuracy: {acc}", end=" ")

    with torch.no_grad():
        correct, total = 0, 0
        running_loss = 0.0

        for x, y in test_loader:
            softmax, decoded = model(x)

            _, predictions = torch.max(softmax, 1)
            total += x.size(0)
            correct += (predictions == y).sum().item()

            c = c_loss(softmax, y)
            a = a_loss(decoded, x)
            t = a + c
            running_loss += t.item()

        running_loss /= len(train_loader)
        acc = 100.0 * correct / total
        history["test_loss"].append(running_loss)
        history["test_acc"].append(acc)

        print(f"Testing Loss: {running_loss} Accuracy: {acc}")
