################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from copy import deepcopy
from torch.nn.modules import batchnorm
from tqdm.auto import tqdm
from mlp_pytorch import MLP
import cifar10_utils

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

def accuracy(predictions, labels):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # correct = np.argmax(predictions) == labels
    # accuracy = np.sum(correct)/len(labels)
    accuracy = (predictions == labels).sum().item()/len(labels)

    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.
    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.
    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    input_size = 3*32*32
    accuracies = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = np.reshape(images, newshape = (len(images), input_size))
            # predictions = model(images)
            # accuracies += accuracy(predictions, labels)
            # total += 1
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            accuracies += accuracy(predicted, labels)
            total += 1

    avg_accuracy = accuracies/total

    #######################
    # END OF YOUR CODE    #
    #######################
    return avg_accuracy


def train(hidden_dims, lr, use_batch_norm, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      use_batch_norm: If True, adds batch normalization layer into the network.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation.
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')
    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Set default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=False)

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    input_size = 3*32*32
    num_classes = 10
    train_loader = cifar10_loader['train']
    valid_loader = cifar10_loader['validation']
    test_loader = cifar10_loader['test']

    # TODO: Initialize model and loss module
    model = MLP(input_size, hidden_dims, num_classes, use_batch_norm)
    loss_module = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr)

    # TODO: Training loop including validation
    train_loss = np.zeros(epochs)
    val_loss =  np.zeros(epochs)
    train_accuracies = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)
    best_i = 0
    best_model = deepcopy(model)

    for epoch in range(epochs): 
        model.train()
        train_running_loss = 0.0

    # TODO: Do optimization with the simple SGD optimizer
        for images, labels in train_loader:
            images = np.reshape(images, newshape = (len(images), input_size))
            opt.zero_grad()

            with torch.set_grad_enabled(True):
                # forward prop
                pred = model(images)
                loss = loss_module(pred, labels)

            # backward prop
            loss.backward()
            opt.step()

            train_running_loss += loss.item()

        train_loss[epoch] = train_running_loss/len(train_loader)
        train_accuracies[epoch] = evaluate_model(model, train_loader)
        
        model.eval()
        val_running_loss = 0.0
        for images, labels in valid_loader:
            images = np.reshape(images, newshape = (len(images), input_size))
            opt.zero_grad()

            with torch.no_grad():
                pred = model(images)
                loss = loss_module(pred, labels)

            val_running_loss += loss.item()

        val_loss[epoch] = val_running_loss/len(valid_loader)
        val_accuracies[epoch] = evaluate_model(model, valid_loader)

        if(val_accuracies[epoch] > val_accuracies[best_i]):
            best_i = epoch
            best_model = deepcopy(model)
    
    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, test_loader)

    # TODO: Add any information you might want to save for plotting
    logging_info = {"train_loss": train_loss, 'val_loss': val_loss, 'train_acc': train_accuracies}

    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_info

def plot(model, val_accuracies, test_accuracy, logging_info):
    train_loss = logging_info['train_loss']
    val_loss = logging_info['val_loss']
    train_acc = logging_info['train_acc']

    plt.plot(train_loss, label = "Training Loss")
    plt.plot(val_loss, label = "Validation Loss")
    plt.legend()
    plt.title("Epoch Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    plt.close()

    plt.plot(train_acc, label = "Training Accuracy")
    plt.plot(val_accuracies, label = "Validation Accuracy")
    plt.legend()
    plt.title("Model Accuracies")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()
    plt.close()  

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    parser.add_argument('--use_batch_norm', action='store_true',
                        help='Use this option to add Batch Normalization layers to the MLP.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    # mod, vals, test_acc, logs = train(**kwargs)
    mod, vals, test_acc, logs = train(hidden_dims=[128], lr=.1, use_batch_norm=True, batch_size=128, epochs=1, seed=42, data_dir='data/')
    print(mod, vals, test_acc)
    plot(mod, vals, test_acc, logs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    