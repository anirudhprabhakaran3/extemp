import torch.nn as nn


class ExTempConvLG(nn.Module):
    def __init__(self):
        super(ExTempConvLG, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=2, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.linear = nn.Sequential(
            nn.Linear(2880, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 4),
            nn.Softmax(1),
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        assert len(x.shape) == 3
        x = self.conv_block1(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
