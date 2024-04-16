import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-3
NUM_EPOCHS = 1000
