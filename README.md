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
Upon successful execution, "Success" will be printed, and four files will be generated: X_train.csv, y_train, X_test.csv, and y_test.csv.

Tips： Please note the modification of the file path
### Step 2: Model Training
Train your own Cropformer.

You need to provide four input files: X_train.csv, y_train.csv, X_test.csv, and y_test.csv.

```bash
python model.py
```

The model file is obtained after training is complete.

Tips： Please note the modification of the file path. Test files are available in the data folder.

### Step 3: Extracting Attentional Weights

Preparing your model file.

```bash
python Attention_weight.py
```

The corresponding weight values are obtained after the run completes.

Tips：Please note the modification of the file path. Test files are available in the data folder.
### Tips
Any general-purpose computer that supports PyTorch can install this software, including systems such as Windows 10+, and Linux. On a Linux computer with 32GB of memory and a 16-core processor, the installation time for this software is less than 10 minutes. The running time is closely related to the dataset; generally, on computers with GPU support, training can be completed within 1 hour using datasets of a not too large scale.
