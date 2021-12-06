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
    line = results['val_loss']

    plt.plot(line)
        
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("MLP Validation Loss Curve")
    #fig.set_tight_layout(True)
    plt.show()
    
if __name__ == '__main__':
    grpah = plot('mlp.txt')