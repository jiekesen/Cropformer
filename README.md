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
python main.py
```
#### Output

- model.pth
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
