## Cropformer
Cropformer is a genomic selection method based on the architecture of Convolutional Neural Networks (CNNs) combined with a multi-head self-attention mechanism, which is used for crop phenotype prediction.

![image](https://github.com/user-attachments/assets/1d12ec0b-3342-4348-98a9-e3a8518fe73f)


## System requirements
Python 3.10 / 3.11.

Optional: Hardware accelerator supporting PyTorch.
## Install Cropformer

We provided a pre-packaged Conda environment for directly running Cropformer.

```bash
conda env create -f cropformer.yml
```
## Run Cropformer
We provide a step-by-step guide for running Cropformer.

### Step 1: Data Preprocessing
Convert the Ped file into a model-compatible CSV format. Test files are available in the data folder.

```bash
python ped_to_matrix.py
```
Tips： Please note the modification of the file path
#### Useful options

- DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
- print(f'Device used: {DEVICE}')
- is_scaler = False
- batch_size = 128  # The number of training examples utilized in one iteration. 
- hidden_dim = 128  # The dimensionality of the hidden layer in a neural network. 
- num_attention_heads = 8 # The number of attention heads in a multi-head attention mechanism.	
- kernel_size = 3 #  The size of the convolutional kernel or filter used in a convolutional neural network (CNN).
- best_acc = 0
- LR = 0.001 # Learning rate
- epochs = 300
- hidden_dropout_prob=0.3 # The number of times the learning algorithm will work through the entire training dataset.
- attention_probs_dropout_prob = 0.3 # The probability of dropout in the hidden layers. 
#### Jupyter
A quick implementation Jupyter script, see **cropformer.ipynb**

### Tips
Any general-purpose computer that supports PyTorch can install this software, including systems such as Windows 10+, Linux, and macOS. On a Linux computer with 32GB of memory and a 16-core processor, the installation time for this software is less than 10 minutes. The running time is closely related to the dataset; generally, on computers with GPU support, training can be completed within 1 hour using datasets of a not too large scale.
