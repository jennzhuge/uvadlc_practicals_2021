import numpy as np
import os
import torch
import matplotlib.pyplot as plt

def plot(results_filename):
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
    lossline = results['loss']
    accline = results['accs']

    plt.plot(lossline)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("LSTM Training Loss Curve")
    #fig.set_tight_layout(True)
    plt.show()

    plt.plot(accline)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("LSTM Training Loss Curve")
    plt.show()
    
if __name__ == '__main__':
    grpah = plot('lossAcc.txt')
    plot('lossAcc_sample.txt')