"""
LSTM model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import numpy as np
import torch
import torch.nn as nn


class LSTM(nn.Module):
    # You will need to complete the class init function, and forward function

    def __init__(self, input_size, hidden_size):
        """ Init function for LSTM class
            Args:
                input_size (int): the number of features in the inputs.
                hidden_size (int): the size of the hidden layer
            Returns: 
                None
        """
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        ################################################################################
        # TODO:                                                                        #
        #   Declare LSTM weights and attributes in order specified below to pass GS.   #
        #   You should include weights and biases regarding using nn.Parameter:        #
        #       1) i_t: input gate                                                     #
        #       2) f_t: forget gate                                                    #
        #       3) g_t: cell gate, or the tilded cell state                            #
        #       4) o_t: output gate                                                    #
        #   for each equation above, initialize the weights,biases for input prior     #
        #   to weights, biases for hidden.                                             #
        #   when initializing the weights consider that in forward method you          #
        #   should NOT transpose the weights.                                          #
        #   You also need to include correct activation functions                      #
        ################################################################################
        # i_t: input gate
        self.W_ii = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_ii = nn.Parameter(torch.zeros(hidden_size))
        self.W_hi = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_hi = nn.Parameter(torch.zeros(hidden_size))
        self.sigmoid_it = nn.Sigmoid()
        # f_t: the forget gate
        self.W_if = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_if = nn.Parameter(torch.zeros(hidden_size))
        self.W_hf = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_hf = nn.Parameter(torch.zeros(hidden_size))
        self.sigmoid_ft = nn.Sigmoid()
        # g_t: the cell gate
        self.W_ig = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_ig = nn.Parameter(torch.zeros(hidden_size))
        self.W_hg = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_hg = nn.Parameter(torch.zeros(hidden_size))
        self.tanh_gt = nn.Tanh()
        # o_t: the output gate
        self.W_io = nn.Parameter(torch.randn(input_size, hidden_size))
        self.b_io = nn.Parameter(torch.zeros(hidden_size))
        self.W_ho = nn.Parameter(torch.randn(hidden_size, hidden_size))
        self.b_ho = nn.Parameter(torch.zeros(hidden_size))
        self.sigmoid_ot = nn.Tanh()
        self.tanh_ot = nn.Tanh()
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        self.init_hidden()

    def init_hidden(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x: torch.Tensor):
        """Assumes x is of shape (batch, sequence, feature)"""

        ################################################################################
        # TODO:                                                                        #
        #   Implement the forward pass of LSTM. Please refer to the equations in the   #
        #   corresponding section of jupyter notebook. Iterate through all the time    #
        #   steps and return only the hidden and cell state, h_t and c_t.              #
        #   h_t and c_t should be initialized to zeros.                                #
        #   Note that this time you are also iterating over all of the time steps.     #
        ################################################################################
        h_t, c_t = torch.zeros(x.shape[0], self.hidden_size), torch.zeros(x.shape[0], self.hidden_size)

        for time in range(x.shape[1]):
            x_time = x[:, time, :]
            # forget gate
            f_t = self.sigmoid_ft(x_time @ self.W_if + self.b_if + h_t @ self.W_hf + self.b_hf)
            # input gate
            i_ts = self.sigmoid_it(x_time @ self.W_ii + self.b_ii + h_t @ self.W_hi + self.b_hi)
            i_tt = self.tanh_gt(x_time @ self.W_ig + self.b_ig + h_t @ self.W_hg + self.b_hg)
            # cell state
            c_t = f_t * c_t + i_ts * i_tt
            # output gate
            o_t = self.sigmoid_it(x_time @ self.W_io + self.b_io + h_t @ self.W_ho + self.b_ho)
            # hidden state
            h_t = self.tanh_ot(c_t) * o_t
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return (h_t, c_t)
