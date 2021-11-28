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
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.
    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Run all hyperparameter configurations as requested
    hiddens = [[128], [256, 128], [512, 256, 128]]
    results = []
    for i in range(len(hiddens)):
        bn = train_mlp_pytorch.train(hiddens[i], .1, True, 128, 20, 42, 'data/')
        nobn = train_mlp_pytorch.train(hiddens[i], .1, False, 128, 20, 42, 'data/')
        results.append([bn, nobn])

    # TODO: Save all results in a file with the name 'results_filename'. This can e.g. by a json file
    torch.save(results, results_filename)

    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.
    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots
    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    results = torch.load(results_filename)

    fig, axs = plt.subplots(2, 3)

    for i, pair in enumerate(results):
        bnmod, bnvals, bntest_acc, bnlogs = pair[0]
        bn_train_acc = bnlogs['train_acc']

        mod, vals, test_acc, logs = pair[1]
        train_acc = logs['train_acc']
        
        #print(bn_train_acc, train_acc)

        axs[0][i].plot(bn_train_acc, label = "Train Accuracy w/ BN")
        axs[0][i].plot(train_acc, label = "Train Accuracy w/out BN")
        axs[0][i].legend()
        #axs[0][i].set_title("Training Accuracy per Epoch w/ and w/out BN")
        axs[0][i].set_xlabel("Epochs")
        axs[0][i].set_ylabel("Accuracy")
    
        axs[1][i].plot(bnvals, label = "Valid Accuracy w/ BN")
        axs[1][i].plot(vals, label = "Valid Accuracy w/out BN")
        axs[1][i].legend()
        #axs[1][i].set_title("Training Accuracy per Epoch w/ and w/out BN")
        axs[1][i].set_xlabel("Epochs")
        axs[1][i].set_ylabel("Accuracy")
    
    #fig.set_tight_layout(True)
    plt.show()

    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results.txt' 
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)