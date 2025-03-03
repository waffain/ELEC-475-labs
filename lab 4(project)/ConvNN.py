import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F

class ConvNN(nn.Module):
    # [Previous ConvNN class implementation remains the same]
    def __init__(self, num_classes=21):
        super(ConvNN, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Decoder - Upsampling layers
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.upconv4 = nn.ConvTranspose2d(32, num_classes, kernel_size=4, stride=2, padding=1)
        
        # Batch normalization for decoder
        self.upbn1 = nn.BatchNorm2d(128)
        self.upbn2 = nn.BatchNorm2d(64)
        self.upbn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        input_size = (x.size(2), x.size(3))
        
        # Encoder path
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxPool(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.maxPool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.maxPool(x)
        
        # Decoder path
        x = self.upconv1(x)
        x = self.upbn1(x)
        x = F.relu(x)
        
        x = self.upconv2(x)
        x = self.upbn2(x)
        x = F.relu(x)
        
        x = self.upconv3(x)
        x = self.upbn3(x)
        x = F.relu(x)
        
        x = self.upconv4(x)
        
        # Interpolate to match target size
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x