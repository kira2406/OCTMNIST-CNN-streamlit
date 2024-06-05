import torch
import torch.nn as nn
import torch.nn.functional as F

# Defining our CNN model
class imageCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 4)

    def forward(self, x, add_dropout=False):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        # Conditionally add dropout layer
        if add_dropout:
            x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
