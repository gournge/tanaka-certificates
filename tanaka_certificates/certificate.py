import numpy as np
from torch import nn


class Certificate(nn.Sequential):
    """A neural network used as a stochastic-system certificate."""


class PiecewiseQuadraticCertificate(Certificate):
    """A piecewise-quadratic neural network used as a stochastic-system certificate."""

    def get_relu_network_weights(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return affine weights in the format consumed by cell discovery."""
        return [
            (
                layer.weight.detach().cpu().numpy().copy(),
                layer.bias.detach().cpu().numpy().copy(),
            )
            for layer in self
            if isinstance(layer, nn.Linear)
        ]

    def get_last_layer_piecewise_quadratic_activation(self):
        from tanaka_certificates.nn.last_layer_activation import (
            PiecewiseQuadraticActivation,
        )

        activations = [
            layer for layer in self if isinstance(layer, PiecewiseQuadraticActivation)
        ]
        if len(activations) != 1:
            raise ValueError("certificate must contain exactly one PWQ activation")
        return activations[0].specification
