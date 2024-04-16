import torch
import torch.utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from constants import DEVICE


class TemperatureDataset(Dataset):
    def __init__(self, x, y):
        super(TemperatureDataset, self).__init__()

        x = x.reshape(x.shape[0], 1, x.shape[1])

        self.x = torch.from_numpy(x).type(torch.FloatTensor).to(DEVICE)
        self.y = torch.from_numpy(y).type(torch.LongTensor).to(DEVICE)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_datasets(train_data, train_labels, test_data, test_labels):
    train_dataset = TemperatureDataset(train_data, train_labels)
    test_dataset = TemperatureDataset(test_data, test_labels)

    return train_dataset, test_dataset


def get_dataloaders(train_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=8)

    return train_loader, test_loader


def scaffold_loaders(train_data, train_labels, test_data, test_labels):
    train_dataset, test_dataset = get_datasets(
        train_data, train_labels, test_data, test_labels
    )
    return get_dataloaders(train_dataset, test_dataset)
