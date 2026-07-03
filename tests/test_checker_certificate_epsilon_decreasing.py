import pytest

from tanaka_certificates.checker import CheckerCertificateEpsilonDecreasing
from tanaka_certificates.sde import OrnsteinUhlenbeck1D
from tanaka_certificates.sde.constant import BrownianMotion


def test_checker_certificate_epsilon_decreasing_proves_ou_generator_bound():
    """For f(x)=-x and V'=-1, LV=V'f=x, whose supremum on [-2,-1] is -1.

    Thus LV <= -epsilon holds for epsilon=1, but fails for epsilon=1.1.
    """
    checker = CheckerCertificateEpsilonDecreasing(
        OrnsteinUhlenbeck1D(mean_reversion=1.0, volatility=1.0, long_term_mean=0.0)
    )

    assert checker(-2.0, -1.0, -1.0, 1.0)
    assert not checker(-2.0, -1.0, -1.0, 1.1)


def test_checker_certificate_epsilon_decreasing_handles_constant_drift():
    """Brownian motion has f(x)=0, so LV=V'f=4*0=0 <= -epsilon for epsilon=0."""
    checker = CheckerCertificateEpsilonDecreasing(BrownianMotion())

    assert checker(-2.0, 3.0, 4.0, 0.0)
    assert not checker(-2.0, 3.0, 4.0, 1e-6)


def test_checker_certificate_epsilon_decreasing_rejects_negative_epsilon():
    """The requested decay margin must satisfy epsilon >= 0 by definition."""
    checker = CheckerCertificateEpsilonDecreasing(BrownianMotion())

    with pytest.raises(ValueError, match="epsilon must be nonnegative"):
        checker(-1.0, 1.0, 1.0, -0.1)
