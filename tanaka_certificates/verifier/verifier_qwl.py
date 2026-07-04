"""

Multidimensional piecewise-quadratic verification of certificates for stochastic reach-
avoid problems.

Assume V is piecewise-quadratic. Then the domain K \subset R^n can be partitioned into a
finite number of convex polytopes {K_i} such that V is quadratic on each K_i,
    V(x) = x^T Q_i x + p_i^T x + c_i, for x in K_i.

The checks are as follows:
- sup V(initial) <= alpha
- inf V(unsafe) >= beta
- sup LV(x) <= -epsilon, where x:
    - is in sub-beta basin outside of target and
    - is in the interior of a cell K_i
- (\grad V_i(x) - \grad V_j(x))^T n_{i->j, F_k} <= 0, for x in the interior of each face
    F_k of the facet F_ij between K_i and K_j, where n_{i->j, F_k} is the unit normal
    pointing from K_i into K_j.

Notes:
- LV(x) = (p_i + Q_i x)^T f(x) + 0.5 tr(a(x) Q_i), where a(x) = g(x) g(x)^T
- (\grad V_i(x) - \grad V_j(x))^T n_{i->j, F_k} =
    (p_i - p_j + (Q_i - Q_j) x)^T n_{i->j, F_k}
- given two polytopes K_i and K_j, the facet F_ij between them may have multiple
    pairwise disjoint components F_k, so the normal and the Tanaka condition must be
    checked on each one.

To have an efficient implementation, we need to have a data structure intialized like:

>>> pql = PiecewiseQuadraticLookup(certificate, sde)

to compute the following:

>>> pql.check_smaller_on_hyperrectangle_union(initial, alpha) # <= alpha
>>> pql.check_smaller_on_hyperrectangle_union(unsafe, beta) # <= beta

>>> # <= -epsilon
>>> pql.check_generator_nonpos_intersection_cell_interior_subbeta_basin(
        target,
        epsilon
    )

This does not necessarily mean we have to make methods:

>>> pql.get_cells() # list of cells K_i
>>> pql.get_cell_facets(cell_index_i) # list of facets F_ij
>>> pql.get_cell_boundary_faces(cell_index_i, cell_index_j) # list of faces F_k
>>> pql.get_quadratic_piece(cell_index_i) # Q_i, p_i, c_i

but a baseline implementation is supposed to be done this way.

"""

from tanaka_certificates.certificate import Certificate
from tanaka_certificates.piecewise_lookup import PiecewiseQuadraticLookupBaseline
from tanaka_certificates.ra import ReachAvoidProblem
from tanaka_certificates.sde.base import SDE
from tanaka_certificates.verifier.base import Verifier, VerificationResult


class VerifierPiecewiseQuadratic(Verifier):
    """Placeholder pending a certified multidimensional implementation."""

    def __init__(
        self,
        sde: SDE,
        reach_avoid_problem: ReachAvoidProblem,
        piecewise_lookup: PiecewiseQuadraticLookupBaseline | None = None,
        certificate: Certificate | None = None,
    ):

        super().__init__(sde, certificate, reach_avoid_problem)
        self.piecewise_lookup = piecewise_lookup

    def verify(self):
        raise NotImplementedError(
            "Piecewise-quadratic verification is not implemented yet."
        )
