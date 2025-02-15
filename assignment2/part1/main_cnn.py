###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import numpy as np
import pickle
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms

from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform
from cifar10_utils import get_train_validation_set, get_test_set


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'debug':  # Use this model for debugging
        cnn_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*32*3, num_classes)
            )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
            cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    return cnn_model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model = get_model(model)
    model.to(device)
    num_classes = 10
    
    # Load the datasets
    train_data, val_data = get_train_validation_set(data_dir)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size,
                                          shuffle = True, num_workers = 3)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size,
                                          shuffle = True, num_workers = 3)
    test_data = get_test_set(data_dir)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size,
                                          shuffle = True, num_workers = 3)
    
    # Initialize the optimizers and learning rate scheduler. 
    # We provide a recommend setup, which you are allowed to change if interested.
    loss_module = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)
    
    # Training loop with validation after each epoch. Save the best model, and remember to use the lr scheduler.
    #train_loss = np.zeros(epochs)
   # val_loss =  np.zeros(epochs)
    train_accuracies = np.zeros(epochs)
    val_accuracies = np.zeros(epochs)
    best_i = 0
    best_model = deepcopy(model)

    for epoch in range(epochs): 
        model.train()
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # forward prop
            pred = model(images)
            loss = loss_module(pred, labels)

            # backward prop
            loss.backward()
            optimizer.step()

        scheduler.step()
        #train_accuracies[epoch] = evaluate_model(model, train_loader, device)
        
        model.eval()
        #for images, labels in val_loader:
            #images = images.to(device)
            #labels = labels.to(device)
            #optimizer.zero_grad()

            #with torch.no_grad():
            #pred = model(images)
            #loss = loss_module(pred, labels)

        val_accuracies[epoch] = evaluate_model(model, val_loader, device)

        if(val_accuracies[epoch] > val_accuracies[best_i]):
            best_i = epoch
            best_model = deepcopy(model)
    
    filename = checkpoint_name
    pickle.dump(best_model, open(filename, 'wb'))
    
    # Load best model and return it.
    model = pickle.load(open(filename, 'rb'))
    
    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.
    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    correct = 0
    total = 0
    accuracies = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted = torch.argmax(torch.sigmoid(outputs.data), 1)
            #correct += (predicted == labels).sum().item()

            accuracies.append(((predicted == labels).float()).cpu().mean())
            #total += 1
            #total += labels.size(0)
            
            #batch_accuracy = (torch.argmax(outputs, axis=1) == labels).sum()/outputs.shape[0]
            #accuracies.append(batch_accuracy)

        accuracy = np.mean(accuracies)

        # accuracy = accurac
    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.
    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(seed)
    test_results = {}
    
    plain_data = get_test_set(data_dir)
    plain_loader = torch.utils.data.DataLoader(plain_data, batch_size,
                                          shuffle = True, num_workers = 3)
    test_results['plain'] = evaluate_model(model, plain_loader, device)
    
    corrupt_fns = [gaussian_noise_transform, gaussian_blur_transform, 
                   contrast_transform, jpeg_transform]
    for fn in corrupt_fns:
        sev_accs = np.zeros(5)
        for severity in [1, 2, 3, 4, 5]:
            test_data = get_test_set(data_dir, transforms.Compose([fn(severity)]))
            test_loader = torch.utils.data.DataLoader(test_data, batch_size,
                                          shuffle = True, num_workers = 3)
            sev_accs[severity - 1] = evaluate_model(model, test_loader, device)
        test_results[fn.__name__] = sev_accs
    #######################
    # END OF YOUR CODE    #
    #######################
    return test_results


def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.
    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(seed)
    print(model_name, epochs)
    
    # Check for existing model, if none train
    filename = model_name + '_model2.sav' 
    if not os.path.isfile(filename):
        best_mod = train_model(model_name, lr, batch_size, epochs, data_dir, filename, device)
    else: best_mod = pickle.load(open(filename, 'rb'))
    
    # Test best model
    filename = model_name + 'results2.txt' 
    if not os.path.isfile(filename):
        results = test_model(best_mod, batch_size, data_dir, device, seed)
        torch.save(results, filename)
    else: results = torch.load(filename)
    print(results)
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--model_name', default='debug', type=str,
                        help='Name of the model to train.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
    #main(model_name = 'resnet18', lr = .01, batch_size = 128, epochs = 1, seed = 42, data_dir = 'data/')