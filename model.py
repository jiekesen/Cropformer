import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.nn import MSELoss
from lightning.pytorch import LightningModule
import pandas as pd
from optuna import create_study
from sklearn.model_selection import KFold, train_test_split
import torch.optim as optim
import os
import functools

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
    def __init__(self, num_attention_heads, input_size, hidden_size, output_dim=1, kernel_size=3,
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
        self.loss_fn = MSELoss()

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



def objective(trial, x_train, y_train, inner_cv, DEVICE, hidden_dim, output_dim, kernel_size, learning_rate):
    num_attention_heads = trial.suggest_categorical('num_attention_heads', [4, 8])
    attention_probs_dropout_prob = trial.suggest_categorical('attention_probs_dropout_prob', [0.2, 0.5])

    fold_losses = []
    for train_idx, valid_idx in inner_cv.split(x_train):
        x_inner_train, x_inner_valid = x_train[train_idx], x_train[valid_idx]
        y_inner_train, y_inner_valid = y_train[train_idx], y_train[valid_idx]

        scaler = StandardScaler()
        x_inner_train = scaler.fit_transform(x_inner_train)
        x_inner_valid = scaler.transform(x_inner_valid)

        x_inner_train_tensor = torch.from_numpy(x_inner_train).float().to(DEVICE)
        y_inner_train_tensor = torch.from_numpy(y_inner_train).float().to(DEVICE)
        x_inner_valid_tensor = torch.from_numpy(x_inner_valid).float().to(DEVICE)
        y_inner_valid_tensor = torch.from_numpy(y_inner_valid).float().to(DEVICE)

        train_data = TensorDataset(x_inner_train_tensor, y_inner_train_tensor)
        valid_data = TensorDataset(x_inner_valid_tensor, y_inner_valid_tensor)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)

        model = SelfAttention(num_attention_heads, x_inner_train.shape[1], hidden_dim, output_dim,
                              hidden_dropout_prob=0.5, kernel_size=kernel_size,
                              attention_probs_dropout_prob=attention_probs_dropout_prob).to(DEVICE)
        loss_function = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(20):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch.reshape(-1, 1))
                loss.backward()
                optimizer.step()

        model.eval()
        valid_losses = []
        with torch.no_grad():
            for x_batch, y_batch in valid_loader:
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch.reshape(-1, 1))
                valid_losses.append(loss.item())

        fold_losses.append(np.mean(valid_losses))

    return np.mean(fold_losses)


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def run_nested_cv_with_early_stopping(data, label, outer_cv, inner_cv, learning_rate, batch_size, hidden_dim,
                                      output_dim, kernel_size, patience, DEVICE):
    best_corr_coefs = []
    import time
    time_star = time.time()
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(data)):
        x_train, x_test = data[train_idx], data[test_idx]
        y_train, y_test = label[train_idx], label[test_idx]
        objective_with_data = functools.partial(objective, x_train=x_train, y_train=y_train, inner_cv=inner_cv,
                                                DEVICE=DEVICE, hidden_dim=hidden_dim, output_dim=output_dim,
                                                kernel_size=kernel_size, learning_rate=learning_rate)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective_with_data, n_trials=20)
        best_trial = study.best_trial
        num_attention_heads = best_trial.params['num_attention_heads']
        # hidden_dim = best_trial.params['hidden_dim']
        # kernel_size = best_trial.params['kernel_size']
        # hidden_dropout_prob = best_trial.params['hidden_dropout_prob']
        attention_probs_dropout_prob = best_trial.params['attention_probs_dropout_prob']

        model = SelfAttention(num_attention_heads, x_train.shape[1], hidden_dim, output_dim,
                              hidden_dropout_prob=hidden_dropout_prob, kernel_size=kernel_size,
                              attention_probs_dropout_prob=attention_probs_dropout_prob).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        x_train_tensor = torch.from_numpy(x_train).float().to(DEVICE)
        y_train_tensor = torch.from_numpy(y_train).float().to(DEVICE)
        x_test_tensor = torch.from_numpy(x_test).float().to(DEVICE)
        y_test_tensor = torch.from_numpy(y_test).float().to(DEVICE)

        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size, shuffle=False)

        early_stopping = EarlyStopping(patience=patience)
        best_corr_coef = -float('inf')
        for epoch in range(100):
            model.train()
            for x_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = model(x_batch)
                loss = loss_function(y_pred, y_batch.reshape(-1, 1))
                loss.backward()
                optimizer.step()

            model.eval()
            y_test_preds, y_test_trues = [], []
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    y_test_pred = model(x_batch)
                    y_test_preds.extend(y_test_pred.cpu().numpy().reshape(-1).tolist())
                    y_test_trues.extend(y_batch.cpu().numpy().reshape(-1).tolist())

            corr_coef = np.corrcoef(y_test_preds, y_test_trues)[0, 1]
            scheduler.step(-corr_coef)

            if corr_coef > best_corr_coef:
                best_corr_coef = corr_coef
                torch.save(model.state_dict(), f'best_model_fold_{fold + 1}.pth')

            early_stopping(corr_coef)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        best_corr_coefs.append(best_corr_coef)

        print(
            f'Fold {fold + 1}: Correlation Coefficient: {corr_coef:.4f}, Best Correlation Coefficient: {best_corr_coef:.4f}')
    average_corr_coef = np.mean(best_corr_coefs)
    print(f"average_corr_coef {average_corr_coef:.4f}")
    time_end = time.time()
    execution_time = int(time_end - time_star)
    data = {'time': [execution_time],
            'mean': [average_corr_coef]}
    pd.DataFrame(data).to_csv("3k_DTH_result_nestcv.csv")

def data_preprocessing(data_path, label_path, target_columns=10000):

    label = pd.read_csv(label_path).values
    data = pd.read_csv(data_path)

    if data.shape[1] < target_columns:
        missing_columns = target_columns - data.shape[1]
        zeros_df = pd.DataFrame(np.zeros((data.shape[0], missing_columns)))
        data = pd.concat([data, zeros_df], axis=1)

    data = data.values
    return data, label


if __name__ == '__main__':
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    batch_size = 128
    hidden_size = 128
    output_dim = 1
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

    data_path = "./X_train.csv"
    label_path = "./y_train.csv"
    data, label = data_preprocessing(data_path, label_path)
    run_nested_cv_with_early_stopping(data=data,
                                      label=label,
                                      outer_cv=outer_cv,
                                      inner_cv=inner_cv,
                                      learning_rate=0.001,
                                      batch_size=32,
                                      hidden_dim=64,
                                      output_dim=1,
                                      kernel_size=3,
                                      patience=5,
                                      DEVICE='cuda' if torch.cuda.is_available() else 'cpu')


    print("successfully")
