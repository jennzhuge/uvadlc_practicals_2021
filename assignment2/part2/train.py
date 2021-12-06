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
# Date Adapted: 2021-11-11
###############################################################################

from datetime import datetime
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel


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


def train(args):
    """
    Trains an LSTM model on a text dataset
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)
    
    # Load dataset, the data loader returns pairs of tensors (input, targets) where inputs 
    # are the input characters, and targets are the labels, i.e. the text shifted by one.
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    args.dataset = dataset
    args.vocabulary_size = dataset._vocabulary_size
    data_loader = DataLoader(dataset, args.batch_size, shuffle=True, drop_last=True, 
                             pin_memory=True, collate_fn=text_collate_fn)
    
    # Debug sample
    # filename = 'lstm_model.sav' 
    # model = pickle.load(open(filename, 'rb'))
    # print(model.sample(temperature = 1))
    # return
    
    # Create model
    model = TextGenerationModel(args)
    model = model.to(args.device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)
    loss_module = nn.CrossEntropyLoss()
    loss_module = loss_module.to(args.device)
    
    # Training loop
    train_loss = np.zeros(args.num_epochs)
    train_accuracies = np.zeros(args.num_epochs)
    for epoch in range(args.num_epochs):
        model.train()
        train_running_loss = 0.0
        accuracies = 0
        #batches = 0

        for chars, labels in data_loader:
            chars = chars.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()

            pred = model(chars)
            pred = pred.permute(0, 2, 1)
            loss = loss_module(pred, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()
            
            train_running_loss += loss.item()
            
            _, predicted = torch.max(pred, 1)
            accuracies += (predicted == labels).float().mean()
            #batches += 1

        train_loss[epoch] = train_running_loss/len(data_loader)
        train_accuracies[epoch] = accuracies/len(data_loader)
        
        # Sample
        sents = []
        if args.sampling:
            if epoch in [0, 5, 19]:
                for temp in [0, .5, 1.0, 2.0]:
                    sentences = model.sample(temperature = temp)
                    sents.append(sentence)
                    print(sentences)
        print(epoch)
    
    # save model
    filename = 'lstm_model.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    # save loss and accuracy
    results = {'loss': train_loss, 'accs': train_accuracies, 'sentences': sents}
    print(results)
    torch.save(results, 'lossAcc.txt')
        
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    #parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')
    #parser.add_argument('--sampling', type=bool, default=False, help='True if we want to print samples')
    parser.add_argument('--sampling', type=bool, default=False, help='True if we want to print samples')

    args = parser.parse_args()
    #args.device = "cpu"
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    #args.sampling = True
    train(args)

