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
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """

    def __init__(self, n_inputs, n_hidden, n_outputs):
        """
        Initializes MLP object.
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_outputs: This number is required in order to specify the
                     output dimensions of the MLP
        TODO: 
        - define a simple MLP that operates on properly formatted QM9 data
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()

        self.layers = nn.ModuleList()
        n_in = n_inputs
        for n_out in n_hidden:
            self.layers.append(nn.Linear(n_in, n_out))
            self.layers.append(nn.ReLU())
            n_in = n_out
        self.layers.append(nn.Linear(n_in, n_outputs))
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        Args:
            x: input to the network
        Returns:
            out: outputs of the network
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = x
        for layer in self.layers:
            out = layer(out)
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device


class GNN(nn.Module):
    """
    Implements a graphical neural network in pytorch. In particular,
    we will use pytorch geometric's nn_conv module so we can apply a neural network to the edges.
    """

    def __init__(self, n_node_features: int, n_edge_features: int,
                 n_hidden: int, n_output: int, num_convolution_blocks: int) -> None:
        """
        Create the gnn
        Args:
            n_node_features: input features on each node
            n_edge_features: input features on each edge
            n_hidden: hidden features within the neural networks 
                (embeddings, nodes after graph convolutions, etc.)
            n_output: how many output features
            num_convolution_blocks: how many blocks convolutions should be performed. 
                A block may include multiple convolutions
        
        TODO: 
        - define a GNN which has the following structure: 
            node embedding -> [ReLU -> RGCNConv -> ReLU -> MFConv] x num_convs -> Add-Pool -> Linear -> ReLU -> Linear
        - Once the data has been pooled, it may be beneficial to apply another MLP on the pooled data before predicing the output.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        super().__init__()
        
        self.n_node_fts = n_node_features
        self.n_edge_fts = n_edge_features
        self.n_hid = n_hidden
        self.n_out = n_output
        self.n_blocks = num_convolution_blocks
        
        layers = []
        layers.append((nn.Linear(n_node_features, n_hidden), 'x -> x' ))
        for conv_layer in range(num_convolution_blocks):
            layers.append(nn.ReLU())
            layers.append((geom_nn.RGCNConv(n_hidden, n_hidden, n_edge_features), 'x, edge_index, edge_type -> x'))
            layers.append(nn.ReLU())
            layers.append((geom_nn.MFConv(n_hidden, n_hidden), 'x, edge_index -> x'))
        self.before_pool = geom_nn.Sequential('x, edge_index, edge_type', layers)
        
        layers2 = [nn.Linear(n_hidden, n_hidden), nn.ReLU(), nn.Linear(n_hidden, n_output)]
        self.after_pool = nn.Sequential(*layers2)
        
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.to(self.device)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)
            edge_attr: edge attributes (pytorch geometric notation)
            batch_idx: Index of batch element for each node
        Returns:
            prediction

        TODO: implement the forward pass being careful to apply MLPs only where they are allowed!
        Hint: remember to use global pooling.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = self.before_pool(x, edge_index, edge_attr)         
        out = geom_nn.global_add_pool(out, batch_idx)                            
        out = self.after_pool(out)
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    @property
    def device(self):
        """
        Returns the device on which the model is. Can be useful in some situations.
        """
        return next(self.parameters()).device
