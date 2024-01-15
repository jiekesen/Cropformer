import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import torch

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


def load_and_process_data(is_scaler, path_xtrain, path_xtest, path_ytrain, path_ytest):

    x_train_reduction = pd.read_csv(path_xtrain)
    x_test_reduction = pd.read_csv(path_xtest)
    y_train = pd.read_csv(path_ytrain)
    y_test = pd.read_csv(path_ytest)


    x_train_reduction = x_train_reduction.values.astype(float)
    x_test_reduction = x_test_reduction.values.astype(float)
    y_train = y_train.values.astype(float)
    y_test = y_test.values.astype(float)


    if is_scaler:
        scaler = StandardScaler()
        scaler_model = StandardScaler()
        scaler.fit(x_train_reduction)
        x_train_reduction = scaler.transform(x_train_reduction)
        x_test_reduction = scaler.transform(x_test_reduction)

        scaler_model.fit(y_train)
        y_train = scaler_model.transform(y_train)
        y_test = scaler_model.transform(y_test)

    x_train_tensor = torch.from_numpy(x_train_reduction).to(torch.float32).to(DEVICE)
    y_train_tensor = torch.from_numpy(y_train).to(torch.float32).to(DEVICE)
    x_test_tensor = torch.from_numpy(x_test_reduction).to(torch.float32).to(DEVICE)
    y_test_tensor = torch.from_numpy(y_test).to(torch.float32).to(DEVICE)

    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size,
                                               True)

    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size,
                                              False)

    return x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor
