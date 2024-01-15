import torch.nn as nn
import torch
import numpy as np

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

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
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
        input_size = input_size - kernel_size + 1
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))


        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = hidden_size


        self.query = nn.Linear(input_size, self.all_head_size)
        self.key = nn.Linear(input_size, self.all_head_size)
        self.value = nn.Linear(input_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attention_probs_dropout_prob)

        self.dense = nn.Linear(hidden_size, input_size)
        self.LayerNorm = LayerNorm(input_size, eps=1e-12)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)
        self.relu = nn.ReLU()
        self.out = nn.Linear(input_size, output_dim)
        self.cnn = nn.Conv1d(1, 1, kernel_size, stride=1)
        self.sigmoid = nn.Sigmoid()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor):
        cnn_hidden = self.cnn(input_tensor.view(input_tensor.size(0), 1, -1))
        # input_tensor = input_tensor.view(input_tensor.size(0), -1, input_tensor.size(-1))
        input_tensor = cnn_hidden

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / np.sqrt(self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(*new_context_layer_shape)

        hidden_states = self.dense(context_layer)

        hidden_states = self.out_dropout(hidden_states)

        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        output = self.out(self.relu(hidden_states.view(hidden_states.size(0), -1)))
        # output = self.sigmoid(output)

        return output