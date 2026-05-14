Neural Network Project – MNIST Digit Classification
Problem Description
This project focuses on solving the problem of handwritten digit recognition using a Multilayer Perceptron (MLP) neural network.
The main goal of the project is to compare the performance of two activation functions:
- ReLU Activation Function
- Sigmoid Activation Function

-Dataset
Dataset Name: MNIST Handwritten Digits
#Source:
MNIST Official Website:
http://yann.lecun.com/exdb/mnist/
or PyTorch MNIST Documentation:
https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html
Description:
The MNIST dataset consists of 70,000 grayscale images of handwritten digits:
60,000 training images
10,000 testing images
Each image is 28×28 pixels

# Google Colab Notebook
[Open in Colab]((https://colab.research.google.com/drive/1Low-1TnJnF24EBNO_HGgVHiVzCZCcUYi?usp=sharing))

The notebook includes:
- Data preprocessing
- Dataset splitting
- Building neural network models
- Training and validation
- Model evaluation
- Accuracy and loss visualization
- Final comparison between both experiments
- The dataset is downloaded automatically using:
python
torchvision.datasets.MNIST(download=True)

# Libraries Used
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Data Preprocessing

The preprocessing pipeline includes:

# 1. Convert Images to Tensor

transforms.ToTensor()


This converts image pixels into tensors.
# 2. Normalize Images
transforms.Normalize((0.5,), (0.5,))

Normalization helps the model train faster and improves stability.

# Transform Pipeline

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset Splitting

The training dataset is divided into:
- 80% Training Data
- 20% Validation Data

Using:
random_split(full_train_dataset, [train_size, val_size])

# DataLoader Configuration
batch_size = 64
Three dataloaders are created:
- train_loader
- val_loader
- test_loader

# Experiment 1 — MLP using ReLU
# Model Architecture
Input Layer  : 784 neurons
Hidden Layer : 128 neurons + BatchNorm + ReLU + Dropout
Hidden Layer : 64 neurons + ReLU
Output Layer : 10 neurons


# ReLU Model

class MLP_ReLU(nn.Module):

# Techniques Used
- Batch Normalization
- Dropout (0.2)
- ReLU Activation

# Optimizer

optim.Adam(model_relu.parameters(), lr=0.001)


# Loss Function
nn.CrossEntropyLoss()

# Experiment 2 — MLP using Sigmoid

# Model Architecture

Input Layer  : 784 neurons
Hidden Layer : 128 neurons + BatchNorm + Sigmoid + Dropout
Hidden Layer : 64 neurons + Sigmoid
Output Layer : 10 neurons


# Sigmoid Model

class MLP_Sigmoid(nn.Module):


# Techniques Used
- Batch Normalization
- Dropout (0.2)
- Sigmoid Activation

# Optimizer

python
optim.Adam(model_sigmoid.parameters(), lr=0.001)


# Training Configuration

python
epochs = 10
batch_size = 64
learning_rate = 0.001


During training, the notebook stores:
- Training Loss
- Validation Loss
- Training Accuracy
- Validation Accuracy

# Evaluation Phase

Both models are evaluated on the test dataset using:
python
model.eval()


Metrics calculated:
- Test Accuracy
- Final Loss

## Visualization

The notebook plots:

# 1. Loss Curves
- ReLU Train Loss
- ReLU Validation Loss
- Sigmoid Train Loss
- Sigmoid Validation Loss

# 2. Accuracy Curves
- ReLU Train Accuracy
- ReLU Validation Accuracy
- Sigmoid Train Accuracy
- Sigmoid Validation Accuracy

Using Matplotlib.


## Final Comparison

At the end of the notebook, both models are compared based on:
- Accuracy
- Final Loss

The comparison helps analyze how activation functions affect neural network performance.
# Key Concepts Used
- Multi-Layer Perceptron (MLP)
- ReLU Activation
- Sigmoid Activation
- Batch Normalization
- Dropout
- Cross Entropy Loss
- Adam Optimizer
- Training & Validation
- Deep Learning using PyTorch

# How to Run the Project

1. Open the notebook in Google Colab or Jupyter Notebook

2. Install required libraries if needed:

pip install torch torchvision matplotlib

3. Run all notebook cells sequentially

# Future Improvements
- Add CNN models for comparison
- Try different activation functions
- Add confusion matrix visualization
- Use GPU training
- Tune hyperparameters




