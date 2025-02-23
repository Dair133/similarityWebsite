from typing import Optional, Dict, Any
import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import KFold
from tqdm import tqdm
import logging
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR  # Import OneCycleLR
import argparse


class MemoryBank:
    def __init__(self, size=4096, feature_dim=64, device='cpu'):
        self.size = size
        self.feature_dim = feature_dim
        self.device = device
        self.memory = F.normalize(torch.randn(size, feature_dim, device=device), dim=1)
        self.ptr = 0

    def update(self, features):
        with torch.no_grad():
            batch_size = features.size(0)
            features = F.normalize(features, dim=1)
            
            # Handle wraparound
            end_ptr = min(self.ptr + batch_size, self.size)
            num_to_update = end_ptr - self.ptr
            
            self.memory[self.ptr:end_ptr] = features[:num_to_update]
            
            # If we have more features and need to wrap around
            remaining = batch_size - num_to_update
            if remaining > 0:
                self.memory[:remaining] = features[num_to_update:]
                
            self.ptr = (self.ptr + batch_size) % self.size

    def get_memory(self):
        return self.memory