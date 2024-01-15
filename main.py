import torch
import torch.nn as nn
from models import SelfAttention
from data_preprocessing import load_and_process_data
from train_test import train_and_test
import config
import torch.nn as nn
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

def main():

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


    x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor = load_and_process_data(
        is_scaler=config.is_scaler,
        path_xtrain="/mnt/win/wanghao/maiz/cnn_tran/class/dtt/ori/xtrain_ori.csv",
        path_xtest="/mnt/win/wanghao/maiz/cnn_tran/class/dtt/ori/xtest_ori.csv",
        path_ytrain="/mnt/win/wanghao/maiz/cnn_tran/class/dtt/ori/ytrain_ori.csv",
        path_ytest="/mnt/win/wanghao/maiz/cnn_tran/class/dtt/ori/ytest_ori.csv",

    )



    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)


    model = SelfAttention(
        num_attention_heads=config.num_attention_heads,
        input_size=x_train_tensor.shape[1],
        hidden_size=config.hidden_dim,
        output_dim=config.output_dim,
        kernel_size=config.kernel_size,
        hidden_dropout_prob=config.hidden_dropout_prob,
        attention_probs_dropout_prob=config.attention_probs_dropout_prob
    ).to(DEVICE)


    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)


    train_and_test(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=loss_function,
        optimizer=optimizer,
        epochs=config.epochs,
        device=DEVICE
    )

if __name__ == "__main__":
    main()
