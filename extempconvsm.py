import torch.nn as nn


class ExTempConvSM(nn.Module):
    def __init__(self):
        super(ExTempConvSM, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=2, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=2, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=2, stride=2),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.linear = nn.Sequential(
            nn.Linear(704, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16, 4),
            nn.Softmax(1),
        )

        self.flatten = nn.Flatten()

    def forward(self, x):
        assert len(x.shape) == 3

        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
