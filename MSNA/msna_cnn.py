import torch
import torch.nn as nn
import torch.nn.functional as F

class MSNA_CNN(nn.Module):
    def __init__(self, n, kernel_size=5, dropout_prob=0.5, fc_dim=32):
        super(MSNA_CNN, self).__init__()

        # Block 1: (2 -> 8 channels)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=8, kernel_size=kernel_size, padding='same')
        self.bn1   = nn.BatchNorm1d(8)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Block 2: (8 -> 16 channels)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=kernel_size, padding='same')
        self.bn2   = nn.BatchNorm1d(16)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # Block 3: (16 -> 32 channels)
        # You can add a third or even a fourth block if you want more capacity
        self.conv3 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernel_size, padding='same')
        self.bn3   = nn.BatchNorm1d(32)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=kernel_size, padding='same')
        self.bn4  = nn.BatchNorm1d(64)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        fc_input_dim = (n * 64) // 16

        # Dropout + Fully connected layers
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1     = nn.Linear(fc_input_dim, fc_dim*2)
        self.fc2     = nn.Linear(fc_dim*2, fc_dim)
        self.fc3     = nn.Linear(fc_dim, 1)

    def forward(self, x):
        """
        x shape: (batch_size, 2, input_length)
        """
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)

        # x = F.relu(self.bn4(self.conv4(x)))
        # x = self.pool4(x)

        # Flatten
        x = x.view(x.size(0), -1)  # shape: (batch_size, 64 * (input_length // 8))
        x = self.dropout(x)

        # Fully connected head
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # Sigmoid for binary classification (peak or no peak)
        return torch.sigmoid(x)

        