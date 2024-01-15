import torch
import torch.nn as nn
from models import SelfAttention
from data_preprocessing import load_and_process_data
from train_test import train_and_test
import config
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used: {DEVICE}')
is_scaler = False
batch_size = 128  
hidden_dim = 128  
output_dim = 3  
num_attention_heads = 8
kernel_size = 3
best_acc = 0
LR = 0.001
epochs = 300
hidden_dropout_prob=0.3
attention_probs_dropout_prob = 0.3