## Cropformer
Cropformer is a genomic selection method based on the architecture of Convolutional Neural Networks (CNNs) combined with a multi-head self-attention mechanism, which is used for crop phenotype prediction.

You can also install the dependent packages by the following commands: 
```bash
pip install python==3.8
pip install numpy
pip install scipy
pip install scikit-learn
pip install pillow
pip install h5py 
pip3 install torch torchvision torchaudio
pip install pandas
pip install requests
```
## Quick Start
#### Run
```bash
To easily reproduce the Cropformer results, run the jupyter file directly
```
#### Processed data
Due to data size limitation issues, the datasets are available at www.a
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
