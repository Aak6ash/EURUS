# Description : An Echo State Network module.

"""
Created on 26 January 2018
@author: Nils Schaetti
Modified on 16 July 2019
@modification by: Aakash
"""


# Imports
import torch
import torch.nn as nn
import ESNcell
import RRcell


# Echo State Network module
class ESN(nn.Module):
    """
    Echo State Network module
    """

    # Constructor
    def __init__(self, input_dim, hidden_dim, output_dim, spectral_radius=0.9, bias_scaling=0, input_scaling=1.0,
                 w=None, w_in=None, w_bias=None, w_fdb=None, sparsity=None, input_set=[1.0, -1.0], w_sparsity=None,
                 nonlin_func=torch.tanh, learning_algo='grad', ridge_param=0.0, create_cell=True,
                 feedbacks=False, with_bias=True, wfdb_sparsity=None, normalize_feedbacks=False,
                 softmax_output=False, seed=None, washout=0, w_distrib='uniform', win_distrib='uniform',
                 wbias_distrib='uniform', win_normal=(0.0, 1.0), w_normal=(0.0, 1.0), wbias_normal=(0.0, 1.0),
                 dtype=torch.float32):
        """
        Constructor
        :param input_dim: Inputs dimension.
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Reservoir size
        :param spectral_radius: Reservoir's spectral radius
        :param bias_scaling: Scaling of the bias, a constant input to each neuron (default: 0, no bias)
        :param input_scaling: Scaling of the input weight matrix, default 1.
        :param w: Internation weights matrix
        :param w_in: Input-reservoir weights matrix
        :param w_bias: Bias weights matrix
        :param w_fdb: Feedback weights matrix
        :param sparsity:
        :param input_set:
        :param w_sparsity:
        :param nonlin_func: Reservoir's activation function (tanh, sig, relu)
        :param learning_algo: Which learning algorithm to use (inv, LU, grad)
        """
        super(ESN, self).__init__()

        # Properties
        self.output_dim = output_dim
        self.feedbacks = feedbacks
        self.with_bias = with_bias
        self.normalize_feedbacks = normalize_feedbacks
        self.washout = washout
        self.dtype = dtype

        # Recurrent layer
        if create_cell:
            self.esn_cell = ESNcell.ESNCell(input_dim, hidden_dim, spectral_radius, bias_scaling, input_scaling, w, w_in,
                                    w_bias, w_fdb, sparsity, input_set, w_sparsity, nonlin_func, feedbacks, output_dim,
                                    wfdb_sparsity, normalize_feedbacks, seed, w_distrib, win_distrib, wbias_distrib,
                                    win_normal, w_normal, wbias_normal, dtype)
        # end if

        # Ouput layer
        self.output = RRcell.RRCell(hidden_dim, output_dim, ridge_param, feedbacks, with_bias, learning_algo, softmax_output, dtype)
    # end __init__

    ###############################################
    # PROPERTIES
    ###############################################

    # Hidden layer
    @property
    def hidden(self):
        """
        Hidden layer
        :return:
        """
        return self.esn_cell.hidden
    # end hidden

    # Hidden weight matrix
    @property
    def w(self):
        """
        Hidden weight matrix
        :return:
        """
        return self.esn_cell.w
    # end w

    # Input matrix
    @property
    def w_in(self):
        """
        Input matrix
        :return:
        """
        return self.esn_cell.w_in
    # end w_in

    ###############################################
    # PUBLIC
    ###############################################

    # Reset learning
    def reset(self):
        """
        Reset learning
        :return:
        """
        # Reset output layer
        self.output.reset()

        # Training mode again
        self.train(False)
    # end reset

    # Output matrix
    def get_w_out(self):
        """
        Output matrix
        :return:
        """
        return self.output.w_out
    # end get_w_out

    # Set W
    def set_w(self, w):
        """
        Set W
        :param w:
        :return:
        """
        self.esn_cell.w = w
    # end set_w

    # forward
    def forward(self, u, y=None, reset_state=False):
        """
        Forward
        :param u: Input signal.
        :param y: Target outputs
        :return: Output or hidden states
        """
        # Compute hidden states

        hidden_states = self.esn_cell(u, reset_state=reset_state)

        # end if

        # Learning algo
        return self.output(hidden_states[:, self.washout:], y)
        # end if
    # end forward

    # Reset hidden layer
    def reset_hidden(self):
        """
        Reset hidden layer
        :return:
        """
        self.esn_cell.reset_hidden()
    # end reset_hidden

    # Get W's spectral radius
    def get_spectral_radius(self):
        """
        Get W's spectral radius
        :return: W's spectral radius
        """
        return self.esn_cell.get_spectral_radius()
    # end spectral_radius

# end ESNCell
