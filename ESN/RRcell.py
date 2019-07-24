# Description : An Echo State Network module.

"""
Created on 26 January 2018
@author: Nils Schaetti
Modified on 16 July 2019
@modification by: Aakash
"""


# Imports
import torch.sparse
import torch
import torch.nn as nn
from torch.autograd import Variable
import math

# Ridge Regression cell
class RRCell(nn.Module):
    """
    Ridge Regression cell
    """

    # Constructor
    def __init__(self, input_dim, output_dim, ridge_param=0.0, feedbacks=False, with_bias=True, learning_algo='inv', softmax_output=False, averaged=False, dtype=torch.float32):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param output_dim: Reservoir size
        """
        super(RRCell, self).__init__()

        # Properties
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ridge_param = ridge_param
        self.feedbacks = feedbacks
        self.with_bias = with_bias
        self.learning_algo = learning_algo
        self.softmax_output = softmax_output
        self.softmax = torch.nn.Softmax(dim=2)
        self.averaged = averaged
        self.n_samples = 0
        self.dtype = dtype

        # Size
        if self.with_bias:
            self.x_size = input_dim + 1
        else:
            self.x_size = input_dim
        # end if

        # Set it as buffer
        self.register_buffer('xTx', Variable(torch.zeros(self.x_size, self.x_size, dtype=dtype), requires_grad=False))
        self.register_buffer('xTy', Variable(torch.zeros(self.x_size, output_dim, dtype=dtype), requires_grad=False))
        self.w_out = nn.Parameter(Variable(torch.zeros(input_dim + 1, output_dim, dtype=dtype)))
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    ###############################################
    # PUBLIC
    ###############################################

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        """self.xTx.data = torch.zeros(self.x_size, self.x_size)
        self.xTy.data = torch.zeros(self.x_size, self.output_dim)
        self.w_out.data = torch.zeros(output_dim, input_dim + 1)"""
        
        self.xTx.data.fill_(0.0)
        self.xTy.data.fill_(0.0)
        # self.w_out.data.fill_(0.0)

        # Training mode again
        self.train(True)
    # end reset

    # Output matrix
    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.w_out
    # end get_w_out

    # Forward
    def forward(self, x, y=None):
        """
        Forward
        :param x: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # x = x.type(torch.FloatTensor)     # x needed to be cast from FloatTensor to DoubleTensor

        # Batch size
        batch_size = x.size()[0]

        # Time length
        time_length = x.size()[1]

        # Add bias
        if self.with_bias:
            x = self._add_constant(x)
        # end if

        # Outputs
        outputs = Variable(torch.zeros(batch_size, time_length, self.output_dim, dtype=self.dtype), requires_grad=False)
        outputs = outputs.cuda() if self.w_out.is_cuda else outputs
            
        # For each batch
        for b in range(batch_size):
            outputs[b] = torch.mm(x[b], self.w_out)
        # end for


        if self.softmax_output:
            return self.softmax(outputs)
        else:
            return outputs
        # end if
    # end forward

    ###############################################
    # PRIVATE
    ###############################################

    # Add constant
    def _add_constant(self, x):
        """
        Add constant
        :param x:
        :return:
        """
        if x.is_cuda:
            bias = Variable(torch.ones((x.size()[0], x.size()[1], 1), dtype=self.dtype).cuda(), requires_grad=False)
        else:
            bias = Variable(torch.ones((x.size()[0], x.size()[1], 1), dtype=self.dtype), requires_grad=False)
        # end if
        return torch.cat((bias, x), dim=2)
    # end _add_constant

# end RRCell
