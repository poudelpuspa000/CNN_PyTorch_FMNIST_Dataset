import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # First Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            32,
            32,
            kernel_size=5,
            stride=2,
            padding=2
        )

        self.bn3 = nn.BatchNorm2d(32)

        self.drop1 = nn.Dropout(0.4)

        # Second Block
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(
            64,
            64,
            kernel_size=5,
            stride=2,
            padding=2
        )

        self.bn6 = nn.BatchNorm2d(64)

        self.drop2 = nn.Dropout(0.4)

        # Third Block
        self.conv7 = nn.Conv2d(64, 128, kernel_size=4)
        self.bn7 = nn.BatchNorm2d(128)

        self.drop3 = nn.Dropout(0.4)

        # Fully Connected Layer
        self.fc = nn.Linear(128, 10)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)

        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.drop2(x)

        x = F.relu(self.bn7(self.conv7(x)))

        x = torch.flatten(x, 1)

        x = self.drop3(x)

        x = self.fc(x)

        return x
