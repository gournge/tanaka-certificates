from tanaka_certificates.piecewise_lookup.base import PiecewiseQuadraticLookup
from tanaka_certificates.piecewise_lookup.cell_discovery import (
    discover_cells_from_network_weights,
    Cell,
)
from tanaka_certificates.certificate import PiecewiseQuadraticCertificate


class PiecewiseQuadraticLookupBaseline(PiecewiseQuadraticLookup):
    """Baseline implementation of PiecewiseQuadraticLookup.

    This implementation is not optimized for performance, but is intended to be
    correct and easy to understand. It is used for testing and verification of
    more optimized implementations.
    """

    def __init__(self, certificate: PiecewiseQuadraticCertificate, sde):
        super().__init__(certificate, sde)

    def get_cells(self) -> list[Cell]:
        """

        The most computationally involved process here.

        We propoagate through the ReLU network (with last layer being a piecewise
        quadratic layer) to extract the quadratic pieces of the certificate. Each piece
        is associated with a cell K_i in the state space, and we return a list of these
        cells.

        Returns:
            list of cells K_i
        """

        return discover_cells_from_network_weights(
            self.certificate.get_relu_network_weights(),
            self.certificate.get_last_layer_piecewise_quadratic_activation(),
        )
