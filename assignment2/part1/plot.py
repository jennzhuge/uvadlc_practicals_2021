import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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
            efs['plain'] = 1 - val
            plt.axhline(val, color = 'k')
        else: 
            efs[key] = val
            line = val
            plt.plot(severity, line, label = key)
        
    plt.legend()
    plt.xlabel("Severity of Corruption")
    plt.xticks(severity)
    plt.ylabel("Accuracy")
    plt.title(model + " Accuracy with Corruption Functions")
    #fig.set_tight_layout(True)
    plt.show()
    
    return efs


def plot_models(resnet18_efs):
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    models = ['vgg11', 'vgg11_bn', 'resnet34', 'densenet121']
    results = []
    for model in models:
        filename = model + 'results2.txt'
        results.append(torch.load(filename))

    severity = [1, 2, 3, 4, 5]
    resnet18_eclean = resnet18_efs['plain']
    print(resnet18_efs['plain'])
    efs = []
    modelces = []
    modelrces = []

    for model in range(len(models)):
        ce = np.zeros(4)
        rce = np.zeros(4)
        for s, (c, val) in enumerate(results[model].items()):
            #print(s, c)
            if c == 'plain':
                efclean = 1 - val
                #efs['efclean'] = efclean
            else: 
                sumefsc = np.sum(1 - val)
                ce[s - 1] = sumefsc/np.sum(1 - resnet18_efs[c])
                #print(ce[s - 1])

                num = np.sum((1 - val) - efclean)
                denom = np.sum((1 - resnet18_efs[c]) - resnet18_eclean)
                print(resnet18_efs[c], 'hi')
                print(denom)
                rce[s - 1] = num/denom
        modelces.append(ce)
        modelrces.append(rce)
    
    corrupt_fns = ['gaussian_noise_transform', 'gaussian_blur_transform', 
                   'contrast_transform', 'jpeg_transform']

    dfce = pd.DataFrame(modelces, index = models)
    dfce.plot.bar()
    plt.xticks(rotation=0)
    plt.legend(corrupt_fns)
    plt.title('CE')
    plt.show()

    dfrce = pd.DataFrame(modelrces, index = models)
    dfrce.plot.bar()
    plt.xticks(rotation=0)
    plt.legend(corrupt_fns)
    plt.title('RCE')
    plt.show()

if __name__ == '__main__':
    #models = ['vgg11', 'vgg11_bn', 'resnet34', 'densenet121']
    resnet18_efs = plot_resnet18('resnet18results2.txt', 'ResNet18')
    #plot_models(resnet18_efs)