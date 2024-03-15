import torch
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used: {DEVICE}')
is_scaler = False
batch_size = 128  
hidden_dim = 128  
output_dim = 1  
num_attention_heads = 8
kernel_size = 3
best_acc = 0
LR = 0.001
epochs = 300
hidden_dropout_prob=0.3
attention_probs_dropout_prob = 0.3



def train_and_test(model, train_loader, test_loader, criterion, optimizer, epochs, device):
    best_corr_coef = -1
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        # train_bar = tqdm(train_loader)  
        for data in train_loader:
            x_train, y_train = data  

            optimizer.zero_grad()
            y_train_pred = model(x_train)
            loss = criterion(y_train_pred, y_train.reshape(-1, 1))
            loss.backward()
            optimizer.step()
            # print(y_train.cpu().detach().numpy().reshape(-1).tolist())
            # print(y_train.shape)

            # corr_coef = np.corrcoef(y_train.cpu().detach().numpy().reshape(-1).tolist(),
            #                         y_train_pred.cpu().detach().numpy().reshape(-1).tolist())[0, 1]
            #
            running_loss += loss.item()
            # train_bar.desc = "train epoch[{}/{}] loss:{:.3f} corr_coef:{:.3f}".format(epoch + 1,
            #                                                                           epochs,
            #                                                                           loss,
            #                                                                           corr_coef)

            # model test
            model.eval()
            y_test_preds, y_test_trues = [], []
            with torch.no_grad():
                # test_bar = tqdm(test_loader)
                for data in test_loader:
                    x_test, y_test = data
                    y_test_pred = model(x_test)

                    y_test_preds.extend(y_test_pred.cpu().numpy().reshape(-1).tolist())
                    y_test_trues.extend(y_test.cpu().numpy().reshape(-1).tolist())
            corr_coef = np.corrcoef(y_test_preds, y_test_trues)[0, 1]
            if corr_coef > best_corr_coef:
                best_corr_coef = corr_coef
                print(f'corr_coef: {best_corr_coef:.3f}, epoch: {epoch + 1}')
                torch.save(model.state_dict(), save_path)

# def train(model, train_loader, criterion, optimizer, device):
#     model.train()
#     total_train = 0
#     correct_train = 0
#     running_loss = 0.0

#     for data in tqdm(train_loader, desc="Training"):
#         inputs, labels = data
#         inputs, labels = inputs.to(device), labels.to(device)

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels.squeeze().long())
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total_train += labels.size(0)
#         correct_train += (predicted == labels.squeeze().long()).sum().item()

#     train_loss = running_loss / len(train_loader)
#     train_accuracy = correct_train / total_train

#     return train_loss, train_accuracy

# def test(model, test_loader, criterion, device):
#     model.eval()
#     total_test = 0
#     correct_test = 0
#     running_loss = 0.0

#     with torch.no_grad():
#         for data in tqdm(test_loader, desc="Testing"):
#             inputs, labels = data
#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs = model(inputs)
#             loss = criterion(outputs, labels.squeeze().long())

#             running_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
    
#             total_test += labels.size(0)
        
#             correct_test += (predicted == labels.squeeze().long()).sum().item()

#     test_loss = running_loss / len(test_loader)
#     test_accuracy = correct_test / total_test

#     return test_loss, test_accuracy

# def train_and_test(model, train_loader, test_loader, criterion, optimizer, epochs, device):
#     best_acc = 0.0

#     for epoch in range(epochs):
        
#         train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
#         test_loss, test_accuracy = test(model, test_loader, criterion, device)
        
#         print(f'Epoch {epoch + 1}/{epochs}')
#         print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')


#         print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

#         if test_accuracy > best_acc:
#             best_acc = test_accuracy
#             print(f'Improved Test Accuracy: {best_acc:.4f} at epoch {epoch + 1}')

#             torch.save(model.state_dict(), "best_model.pth")

#     print('Training and testing complete.')
