import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from lightning.pytorch import LightningModule
import pandas as pd
from optuna import create_study
from sklearn.model_selection import KFold, train_test_split
import torch.optim as optim
import os
import functools
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SelfAttention(LightningModule):
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=3, kernel_size=3,
                 hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5, learning_rate=0.001):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = torch.nn.Linear(input_size, self.all_head_size)
        self.key = torch.nn.Linear(input_size, self.all_head_size)
        self.value = torch.nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = torch.nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.dense = torch.nn.Linear(hidden_size, input_size)
        self.LayerNorm = torch.nn.LayerNorm(input_size, eps=1e-12)
        self.relu = torch.nn.ReLU()
        self.out = torch.nn.Linear(input_size, output_dim)
        self.cnn = torch.nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)

        self.learning_rate = learning_rate
        self.loss_fn = CrossEntropyLoss()

    def forward(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        self.cnn = self.cnn.to(self.device)

        cnn_hidden = self.cnn(input_tensor.view(input_tensor.size(0), 1, -1))

        input_tensor = cnn_hidden
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = mixed_query_layer
        key_layer = mixed_key_layer
        value_layer = mixed_value_layer

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        output = self.out(self.relu(hidden_states.view(hidden_states.size(0), -1)))
        return output

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        val_loss = self.loss_fn(y_pred, y)
        return val_loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate)


if __name__ == '__main__':


    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device used: {DEVICE}')

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    batch_size = 64
    hidden_size = 128
    output_dim = 3
    num_attention_heads = 8
    kernel_size = 3
    best_corr_coef = -1
    best_corr_coefs = 0
    learning_rate = 0.001
    epochs = 100
    input_size = 10000
    hidden_dropout_prob = 0.5
    attention_probs_dropout_prob = 0.5
    hidden_dim = 128
    patience = 5

    data_path = "./class_data/X_train_class.csv"
    label_path = "./class_data/y_train_class.csv"

    data, label = data_preprocessing(data_path, label_path)

    run_nested_cv_with_early_stopping(data=data,
                                      label=label,
                                      outer_cv=outer_cv,
                                      inner_cv=inner_cv,
                                      learning_rate=0.001,
                                      batch_size=32,
                                      hidden_dim=64,
                                      output_dim=3,
                                      kernel_size=3,
                                      patience=5,
                                      DEVICE='cuda' if torch.cuda.is_available() else 'cpu')


    print("successfully")