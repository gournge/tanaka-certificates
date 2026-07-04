from abc import ABC, abstractmethod

from tanaka_certificates.sde.base import SDE
from tanaka_certificates.certificate import Certificate, PiecewiseQuadraticCertificate
from tanaka_certificates.regions import HyperrectangleUnion


class PiecewiseLookup(ABC):

    def __init__(self, certificate: Certificate, sde: SDE):
        self.certificate = certificate
        self.sde = sde

    @abstractmethod
    def check_smaller_on_hyperrectangle_union(
        self, hyperrectangle_union: HyperrectangleUnion, alpha: float
    ) -> bool:
        pass

    @abstractmethod
    def check_generator_nonpos_intersection_cell_interior_subbeta_basin(
        self, target: HyperrectangleUnion, epsilon: float
    ) -> bool:
        pass

    @abstractmethod
    def check_generator_nonpos_intersection_cell_interior_subbeta_basin(
        self, target: HyperrectangleUnion, epsilon: float
    ) -> bool:
        pass


class PiecewiseQuadraticLookup(PiecewiseLookup):
    """A piecewise-quadratic lookup for a piecewise-quadratic certificate."""

    def __init__(self, certificate: PiecewiseQuadraticCertificate, sde: SDE):
        super().__init__(certificate, sde)
