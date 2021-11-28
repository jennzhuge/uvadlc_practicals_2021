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
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization
        Also, initialize gradients with zeros.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Kaiming initialization for weights and zeros for biases
        if(input_layer):
            kaiming = np.sqrt(1/in_features)
        else: 
            kaiming = np.sqrt(2/in_features)
        self.params = {'weight': np.random.normal(loc = 0, scale = kaiming, size = (out_features, in_features)),
                       'bias':   np.zeros(out_features)}

        # Initialize gradients with zeros
        self.grads = {'weight': np.zeros((out_features, in_features)),
                      'bias':   np.zeros(out_features)}

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Y = XW^T + B
        out = x @ self.params['weight'].T + self.params['bias']
        self.prev_x = x
        # print(out.shape)
        # print(self.n_classes)

        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        dx = dout @ self.params['weight']
        # dL/dW = (dL/dY)^T*X
        self.grads['weight'] = dout.T @ self.prev_x
        # dL/db = 1^T(dL/dY) 
        # print(dout.shape, self.grads['weight'].shape)
        self.grads['bias'] = np.ones(dout.shape[0]) @ dout

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.prev_x = None

        #######################
        # END OF YOUR CODE    #
        #######################


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = np.maximum(0, x)
        self.prev_x = out

        #######################
        # END OF YOUR CODE    #
        ####################### 
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # dL/dX = (dL/dY)(ReLu deriv)
        drelu = self.prev_x > 0
        dx = dout*drelu

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.prev_x = None

        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Exp-normalize trick 
        y = np.exp(x - np.max(x, 1, keepdims = True))
        out = y/np.sum(y, 1, keepdims = True)
        self.prev_x = out

        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # dL/dX = Y o [(dL/dY) - ((dL/dY) o Y)1_N*1_N.T]
        oneMat = np.ones((self.prev_x.shape[1], self.prev_x.shape[1]))
        # print(oneMat.shape, self.prev_out.shape, dout.shape)
        dx = self.prev_x*(dout - (dout*self.prev_x) @ oneMat)

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        self.prev_x = None
        
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Li = -(1/S)sum(Tiklog(Xik))
        S = len(x)
        # print(x.shape[1], y)
        # One-hot encode labels
        t = np.eye(x.shape[1])[y]
        #self.t = t
        out = (-1/S)*np.sum(t*np.log(x))

        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # dL/dX = (-1/S)Telemwise/X
        S = len(x)
        #print(x.shape[1], len(y))
        t = np.eye(x.shape[1])[y]
        dx = (-1/S)*(t/x)

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx