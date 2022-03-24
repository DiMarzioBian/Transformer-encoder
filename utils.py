import numpy as np
import torch.nn as nn


def init_embedding(layer):
    """ Initialize embedding """
    bias = np.sqrt(3.0 / layer.embedding_dim)
    nn.init.uniform_(layer.weight, -bias, bias)


def init_linear(input_linear):
    """ Initialize linear layer """
    bias = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -bias, bias)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()
