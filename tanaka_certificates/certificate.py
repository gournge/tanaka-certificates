from torch import nn


class Certificate(nn.Sequential):
    """A neural network used as a stochastic-system certificate."""


class Certificate1D(Certificate):
    """A one-dimensional neural network used as a stochastic-system certificate."""
