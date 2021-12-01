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
import math
import torch
import torch.nn as nn

class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """
    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.
        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################     
        self.Wgx = nn.Parameter(torch.Tensor(lstm_hidden_dim, embedding_size))
        self.Wgh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.bg = nn.Parameter(torch.Tensor(lstm_hidden_dim, 1))
        self.tanh = nn.Tanh()
        
        self.Wix = nn.Parameter(torch.Tensor(lstm_hidden_dim, embedding_size))
        self.Wih = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.bi = nn.Parameter(torch.Tensor(lstm_hidden_dim, 1))
        self.sigmoid = nn.Sigmoid()
        
        self.Wfx = nn.Parameter(torch.Tensor(lstm_hidden_dim, embedding_size))
        self.Wfh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.bf = nn.Parameter(torch.Tensor(lstm_hidden_dim, 1))
        
        self.Wox = nn.Parameter(torch.Tensor(lstm_hidden_dim, embedding_size))
        self.Woh = nn.Parameter(torch.Tensor(lstm_hidden_dim, lstm_hidden_dim))
        self.bo = nn.Parameter(torch.Tensor(lstm_hidden_dim, 1))
        
        # self.device = 'cuda'
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.
        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for param in self.parameters():
            param.data.uniform_(-1/math.sqrt(self.hidden_dim), 1/math.sqrt(self.hidden_dim))
        
        self.bf.data += 1
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds):
        """
        Forward pass of LSTM.
        Args:
            embeds: embedded input sequence with shape [input length, batch size, embedding_size].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        input_len, batch_size, embed_size = embeds.shape
        x = embeds
        self.prev_h = torch.zeros(self.hidden_dim, batch_size).to(self.device)
        self.prev_c = torch.zeros(self.hidden_dim, batch_size).to(self.device)
        hiddens = torch.zeros((input_len, batch_size, self.hidden_dim))
        
        for t in range(input_len):
            # g(𝑡) = tanh(W𝑔𝑥x(𝑡) + W𝑔 ℎh(𝑡−1) + b𝑔)
            y = self.Wgx@x[t, :, :].T
            g = self.tanh(self.Wgx@x[t].T + self.Wgh@self.prev_h + self.bg)
            # i(𝑡) = 𝜎(W𝑖𝑥x(𝑡) + W𝑖 ℎh(𝑡−1) + b𝑖)
            i = self.sigmoid(self.Wix@x[t].T + self.Wih@self.prev_h + self.bi)
            # f(𝑡) = 𝜎(W𝑓 𝑥x(𝑡) + W𝑓 ℎh(𝑡−1) + b𝑓)
            f = self.sigmoid(self.Wfx@x[t].T + self.Wfh@self.prev_h + self.bf)
            # o(𝑡) = 𝜎(W𝑜𝑥x(𝑡) + W𝑜 ℎh(𝑡−1) + b𝑜)
            o = self.sigmoid(self.Wox@x[t].T + self.Woh@self.prev_h + self.bo)
            # c(𝑡) = g(𝑡) ⊙ i(𝑡) + c(𝑡−1) ⊙ f(𝑡)
            c = g*i + self.prev_c*f
            # h(𝑡) = tanh(c(𝑡)) ⊙ o(𝑡)
            h = self.tanh(c)*o
            self.prev_h = h
            self.prev_h = c
            hiddens[t] = h.T
            
        return hiddens
        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """
    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.embed_size = args.embedding_size
        self.vocab_size = args.vocabulary_size
        self.lstm_hidden_dim = args.lstm_hidden_dim
        self.dataset = args.dataset
        self.device = args.device
        
        # self.layers = nn.ModuleList()
        self.embedder = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = LSTM(self.lstm_hidden_dim, self.embed_size)
        # self.lin = nn.Linear(self.lstm_hidden_dim, self.vocab_size).to(args.device)
        self.lin = nn.Linear(self.lstm_hidden_dim, self.vocab_size)
        
        self.to(self.device)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        embeds = self.embedder(x)
        # p(𝑡) = W𝑝 ℎh(𝑡) + b
        p = self.lstm(embeds).to(self.device)
        p = p.to(self.device)
        out = self.lin(p)
        return out 
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=5, sample_length=30, temperature=0.):
        """
        Sampling from the text generation model.
        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        samples = torch.zeros(batch_size)
        for sample in range(batch_size):
            # Set first character randomly
            first_char = torch.randint(0, self.vocab_size)
            sample_str = self.dataset.convert_to_string(first_char)
            
            for char in range(sample_length):
                pred = self.model.forward(samplestr)
                if temp == 0.:
                    nexti = torch.max(pred)
                else: nexti = ''
                sample_str += self.dataset.convert_to_string(nexti)
                
            samples[sample] = sample_str
            
        return samples
        #######################
        # END OF YOUR CODE    #
        #######################
