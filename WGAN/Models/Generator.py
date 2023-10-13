import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg

class Generator32(nn.Module):

    def __init__(self):
        super(Generator32, self).__init__()

        input_size=128+cfg.GAN_NOISE_SIZE
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 32768),
            nn.BatchNorm1d(32768),
        )

        self.conv2 = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=4, stride=1,padding=0),
            nn.BatchNorm3d(512),
        )

        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=3),  # Correct padding
            nn.BatchNorm3d(256),
        )

        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=3, stride=2, padding=3),  # Correct padding
            nn.BatchNorm3d(128),
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(128, 4, kernel_size=4, stride=2, padding=0),  # Correct padding
        )

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = x.view(-1, 512, 4, 4, 4)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x, inplace=True)

        # Conv3
        x = self.conv3(x)
        x = F.relu(x, inplace=True)

        # Conv4
        x = self.conv4(x)
        x = F.relu(x, inplace=True)

        # Conv5
        logits = self.conv5(x)

        sigmoid_output = torch.sigmoid(logits)

        return {'sigmoid_output': sigmoid_output, 'logits': logits}

class Generator32_Small(nn.Module):

    def __init__(self):
        super(Generator32_Small, self).__init__()

        input_size=128+cfg.GAN_NOISE_SIZE
        
        self.fc1 = nn.Sequential(
            nn.Linear(input_size, 8192),
            nn.BatchNorm1d(8192),
        )


        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),  # Correct padding
            nn.BatchNorm3d(256),
        )

        self.conv4 = nn.Sequential(
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),  # Correct padding
            nn.BatchNorm3d(128),
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(64, 4, kernel_size=4, stride=2, padding=1),  # Correct padding
        )

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = x.view(-1, 128, 4, 4, 4)
        
        # Conv2
        x = self.conv2(x)
        x = F.relu(x, inplace=True)

        # Conv3
        x = self.conv3(x)
        x = F.relu(x, inplace=True)

        # Conv4
        x = self.conv4(x)
        x = F.relu(x, inplace=True)

        # Conv5
        logits = self.conv5(x)

        sigmoid_output = torch.sigmoid(logits)

        return {'sigmoid_output': sigmoid_output, 'logits': logits}



