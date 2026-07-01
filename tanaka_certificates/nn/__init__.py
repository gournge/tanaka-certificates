import torch
from torch import nn
from tanaka_certificates.facet import Facet, Breakpoint
from tanaka_certificates.certificate import Certificate


def create_certificate_given_facets(facets: dict[Facet, float]) -> Certificate:
    raise NotImplementedError("This function is not yet implemented.")


def create_1d_certificate_given_breakpoints(
    breakpoints: list[Breakpoint],
    leftmost_slope: float,
    rightmost_slope: float,
) -> Certificate:
    """
    Since every `Breakpoint` just represents a pair

    >>> (x_i, V(x_i)): tuple[float, float]

    it should correspond to a neural network which is piecewise linear with
    breakpoints at x_i and values V(x_i) at those breakpoints.
    Additionally, the neural network should have the specified slopes away
    from the leftmost and rightmost breakpoints.

    >>> c = create_1d_certificate_given_breakpoints(
        breakpoints=[
            Breakpoint(np.array([0.0]), np.array([1.0])),
            Breakpoint(np.array([1.0]), np.array([2.0]))
        ],
        leftmost_slope=0.0,
        rightmost_slope=0.0
    )
    >>> c.forward(torch.tensor([[-1.0], [0.0], [0.5], [1.0], [2.0]]))
    ... tensor([[1.0000],
                [1.0000],
                [1.5000],
                [2.0000],
                [2.0000]])

    Plot:
    >>>       ____
    >>>      /
    >>> ____/

    """
    if not breakpoints:
        raise ValueError("At least one breakpoint is required.")

    points = sorted(
        (
            (float(breakpoint.get_breakpoint), float(breakpoint.get_value))
            for breakpoint in breakpoints
        ),
        key=lambda point: point[0],
    )
    x = torch.tensor([point[0] for point in points])
    y = torch.tensor([point[1] for point in points])

    if len(points) > 1:
        dx = x[1:] - x[:-1]
        if torch.any(dx == 0):
            raise ValueError("Breakpoints must have distinct coordinates.")
        inner_slopes = (y[1:] - y[:-1]) / dx
        right_facing_weights = torch.cat(
            (
                inner_slopes[:1],
                inner_slopes[1:] - inner_slopes[:-1],
                torch.tensor([rightmost_slope]) - inner_slopes[-1:],
            )
        )
    else:
        right_facing_weights = torch.tensor([rightmost_slope])

    hidden = nn.Linear(1, len(points) + 1)
    output = nn.Linear(len(points) + 1, 1)
    with torch.no_grad():
        hidden.weight.fill_(1.0)
        hidden.weight[0, 0] = -1.0
        hidden.bias[0] = x[0]
        hidden.bias[1:] = -x
        output.weight[0, 0] = -leftmost_slope
        output.weight[0, 1:] = right_facing_weights
        output.bias[0] = y[0]

    return Certificate(hidden, nn.ReLU(), output).requires_grad_(False)
