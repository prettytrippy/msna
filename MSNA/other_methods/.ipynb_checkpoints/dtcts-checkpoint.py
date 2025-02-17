import torch
import torch.nn as nn
import torch.nn.functional as F

class DTCT1(nn.Module):
    def __init__(self):
        super(DTCT1, self).__init__()

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(4001, 2000)
        self.fc2 = nn.Linear(2000, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 32)
        self.fc6 = nn.Linear(32, 2)

        self.relu = F.relu
        self.sigmoid = F.sigmoid
        self.softmax = F.softmax
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        # x = self.softmax(x, dim=0)
        return x
        
class DTCT2(nn.Module):
    def __init__(self):
        super(DTCT2, self).__init__()

        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)

        self.relu = F.relu
        self.sigmoid = F.sigmoid
        self.softmax = F.softmax
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        # x = self.softmax(x, dim=0)
        return x

class DTCT3(nn.Module):
    def __init__(self):
        super(DTCT3, self).__init__()

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 2)

        self.relu = F.relu
        self.sigmoid = F.sigmoid
        self.softmax = F.softmax
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.sigmoid(self.fc5(x))
        x = self.sigmoid(self.fc6(x))
        # x = self.softmax(x, dim=0)
        return x