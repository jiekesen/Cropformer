import time
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used: {DEVICE}')

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


class SelfAttention(nn.Module):
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=1, kernel_size=3,
                 hidden_dropout_prob=0.5, attention_probs_dropout_prob=0.5):
        super(SelfAttention, self).__init__()

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention heads (%d)"
                % (hidden_size, num_attention_heads))

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size

        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        # 初始化每个头的重要性评分
        self.head_importance = nn.Parameter(torch.ones(num_attention_heads))

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.dense = nn.Linear(hidden_size, input_size)
        self.LayerNorm = LayerNorm(input_size, eps=1e-12)
        self.relu = nn.ReLU()
        self.out = nn.Linear(input_size, output_dim)
        self.cnn = nn.Conv1d(1, 1, kernel_size, stride=1, padding=1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        input_tensor = input_tensor.view(input_tensor.size(0), -1, input_tensor.size(-1))  # (128,1,10000)

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)


        context_layer = torch.matmul(attention_probs, value_layer)


        head_importance_normalized = self.head_importance / self.head_importance.sum()
        head_weighted_output = context_layer * head_importance_normalized.unsqueeze(-1).unsqueeze(-1)


        self.update_head_importance(attention_probs)

        context_layer = head_weighted_output.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size(0), -1, self.all_head_size)

        hidden_states1 = self.dense(context_layer)
        hidden_states2 = self.out_dropout(hidden_states1)
        hidden_states = self.LayerNorm(hidden_states2 + input_tensor)
        output = self.out(self.relu(hidden_states.view(hidden_states.size(0), -1)))

        return output, hidden_states1

    def update_head_importance(self, attention_probs):

        self.head_importance.data += attention_probs.mean(dim=(0, 1)).sum(dim=-1)


        # for head_idx in range(self.num_attention_heads):
        #     gradient = torch.autograd.grad(loss, self.head_importance[head_idx], retain_graph=True)
        #     self.head_importance.data[head_idx] += gradient.data

def load_model_and_extract_weights(model_path, input_size, hidden_dim, output_dim, num_attention_heads, kernel_size,
                                   hidden_dropout_prob, attention_probs_dropout_prob):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SelfAttention(num_attention_heads, input_size, hidden_dim, output_dim,
                          kernel_size=kernel_size, hidden_dropout_prob=hidden_dropout_prob,
                          attention_probs_dropout_prob=attention_probs_dropout_prob).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=False)
    model.eval()

    return model

def extract_attention_weights(model, x_test_tensor):
    model.eval()
    with torch.no_grad():
        _, attention_probs = model(x_test_tensor)

    print(f"Attention weights shape: {attention_probs.shape}")

    attention_weights = attention_probs.cpu().numpy()
    return attention_weights

def attention_weights_to_dataframe(attention_weights):
    print(f"Attention weights shape: {attention_weights.shape}")

    num_samples, num_heads, seq_len = attention_weights.shape
    attention_list = []

    for sample_idx in range(num_samples):
        for head_idx in range(num_heads):
            for seq_idx in range(seq_len):
                row = {
                    "Sample": sample_idx,
                    "Head": head_idx,
                    "Position": seq_idx,
                    "Weight": attention_weights[sample_idx, head_idx, seq_idx].mean()
                }
                attention_list.append(row)

    attention_df = pd.DataFrame(attention_list)
    return attention_df

def attention_weights_to_avg_feature_weights(attention_weights):
    print(f"Attention weights shape: {attention_weights.shape}")

    avg_feature_weights = attention_weights.mean(axis=(0, 1))
    feature_weight_df = pd.DataFrame({
        "Feature": np.arange(len(avg_feature_weights)),
        "AverageAttentionWeight": avg_feature_weights
    })
    return feature_weight_df

if __name__ == "__main__":

    model_path = './test_model.pth'
    input_size = 10000
    hidden_dim = 64
    output_dim = 1
    num_attention_heads = 8
    kernel_size = 3
    hidden_dropout_prob = 0.5
    attention_probs_dropout_prob = 0.5

    data_path = "./X_train.csv"

    data = pd.read_csv(data_path)

    data = data.values.astype(float)
    data = torch.from_numpy(data).float().to(DEVICE)

    model = load_model_and_extract_weights(model_path,
                                           input_size,
                                           hidden_dim,
                                           output_dim,
                                           num_attention_heads,
                                           kernel_size,
                                           hidden_dropout_prob,
                                           attention_probs_dropout_prob)


    attention_weights = extract_attention_weights(model, data)

    attention_df = attention_weights_to_dataframe(attention_weights)

    avg_feature_weights_df = attention_weights_to_avg_feature_weights(attention_weights)

    avg_feature_weights_df.to_csv("./attention_weights.csv", index=False)

    print("successfully")