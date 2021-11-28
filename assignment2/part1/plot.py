import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def plot_resnet18(results_filename, model):
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
    severity = [1, 2, 3, 4, 5]
    efs = {}

    for key, val in results.items():
        if key == 'plain':
            efs['clean'] = 1 - val
            plt.axhline(val, color = 'k')
        else: 
            efs[key] = val
            plt.plot(severity, val, label = key)
        
    plt.legend()
    plt.xlabel("Severity of Corruption")
    plt.xticks(severity)
    plt.ylabel("Accuracy")
    plt.title(model + " Accuracy with Corruption Functions")
    #fig.set_tight_layout(True)
    plt.show()
    
    return efs

def plot_models(results_filename, model, resnet18_efs):
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
    severity = [1, 2, 3, 4, 5]
    efs = {}

    for key, val in results.items():
        es = {}
        if key == 'plain':
            efclean = 1 - val
            es['efclean'] = [efclean]
        else: 
            rce = 0
            ce = 0
            plt.plot(severity, val, label = key)
        
    plt.legend()
    plt.xlabel("Severity of Corruption")
    plt.xticks(severity)
    plt.ylabel("Accuracy")
    plt.title(model + " Accuracy with Corruption Functions")
    #fig.set_tight_layout(True)
    plt.show()
    
    
if __name__ == '__main__':
    #models = ['vgg11', 'vgg11_bn', 'resnet34', 'densenet121']
    resnet18_efs = plot_resnet18('resnet18results.txt', 'resnet18')
    
    # for model in models:
    #     filename = model + 'results.txt'
    #     plot_models(filename, model, resnet18_efs)