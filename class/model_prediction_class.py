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

    num_attention_heads = 8
    hidden_dropout_prob = 0.5
    kernel_size = 3
    attention_probs_dropout_prob = 0.5
    output_dim = 3
    hidden_size = 64
    input_size = 10000

    model = SelfAttention(num_attention_heads,
                              input_size = input_size,
                              hidden_size = hidden_size ,
                              output_dim = output_dim,
                              hidden_dropout_prob=hidden_dropout_prob,
                              kernel_size=kernel_size,
                              attention_probs_dropout_prob=attention_probs_dropout_prob).to(DEVICE)

    model.load_state_dict(torch.load('./class_model.pth', map_location=torch.device('cpu')))

    X_test = pd.read_csv('./class_data/X_test_class.csv')
    X_test = np.array(X_test)
    X_test_tensor = torch.from_numpy(X_test).to(torch.float32).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(X_test_tensor)
        _, output = torch.max(output.data, 1)
        output = output.cpu().numpy()
    pd.DataFrame(output, columns=['result']).to_csv("./predicted_result.csv")
    print("Successful")