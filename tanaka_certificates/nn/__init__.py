import torch
from torch import nn
from tanaka_certificates.facet import Facet, Breakpoint
from tanaka_certificates.certificate import Certificate


def create_certificate_given_facets(facets: dict[Facet, float]) -> Certificate:
    raise NotImplementedError("This function is not yet implemented.")


def _make_pwl_relu_network(xs, ys, L=2, dtype=torch.float64):
    """
    Construct an L-layer ReLU network representing the continuous piecewise-linear
    function through the knots (xs[i], ys[i]), with linear extrapolation outside
    the knot range.

    Here "L-layer" means L affine/linear layers, with ReLU activations after
    every layer except the final output layer.

    Parameters
    ----------
    xs : array-like, shape (n,)
        Strictly increasing x-coordinates of knots.
    ys : array-like, shape (n,)
        Function values at the knots.
    L : int
        Number of affine layers. Must be >= 2 unless the function is affine.
    dtype : torch dtype
        Numeric dtype.

    Returns
    -------
    net : nn.Sequential
        A ReLU network such that net(x) == V(x) for all x, up to floating-point error.
    """

    xs = torch.tensor(xs, dtype=dtype)
    ys = torch.tensor(ys, dtype=dtype)

    if xs.ndim != 1 or ys.ndim != 1:
        raise ValueError("xs and ys must be one-dimensional.")
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length.")
    if len(xs) < 2:
        raise ValueError("Need at least two knots.")
    if not torch.all(xs[1:] > xs[:-1]):
        raise ValueError("xs must be strictly increasing.")
    if L < 2:
        raise ValueError("This construction requires L >= 2.")

    # Slopes on intervals [xs[i], xs[i+1]]
    slopes = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])

    # Leftmost slope
    s0 = slopes[0]

    # Interior slope jumps.
    # At xs[i], jump = slopes[i] - slopes[i-1]
    interior_knots = xs[1:-1]
    slope_jumps = slopes[1:] - slopes[:-1]

    # Representation:
    # V(x) = bias + s0*x + sum_j slope_jumps[j] ReLU(x - interior_knots[j])
    # with bias chosen so V(xs[0]) = ys[0]
    output_bias = ys[0] - s0 * xs[0]

    # Use x = ReLU(x) - ReLU(-x)
    # Hidden units are:
    # h0 = ReLU(x)
    # h1 = ReLU(-x)
    # h_{j+2} = ReLU(x - interior_knots[j])
    width = 2 + len(interior_knots)

    first = nn.Linear(1, width, bias=True, dtype=dtype)

    W1 = torch.zeros(width, 1, dtype=dtype)
    b1 = torch.zeros(width, dtype=dtype)

    W1[0, 0] = 1.0  # x
    W1[1, 0] = -1.0  # -x

    for j, t in enumerate(interior_knots):
        W1[j + 2, 0] = 1.0
        b1[j + 2] = -t

    first.weight.data.copy_(W1)
    first.bias.data.copy_(b1)

    out_weights = torch.zeros(1, width, dtype=dtype)
    out_weights[0, 0] = s0
    out_weights[0, 1] = -s0

    for j, jump in enumerate(slope_jumps):
        out_weights[0, j + 2] = jump

    layers = [first, nn.ReLU()]

    if L == 2:
        final = nn.Linear(width, 1, bias=True, dtype=dtype)
        final.weight.data.copy_(out_weights)
        final.bias.data.copy_(output_bias.reshape(1))
        layers.append(final)
        return nn.Sequential(*layers)

    # For L > 2, insert extra ReLU layers while preserving the scalar value.
    #
    # First collapse hidden representation to y and -y, then ReLU:
    # [ReLU(y), ReLU(-y)]
    collapse = nn.Linear(width, 2, bias=True, dtype=dtype)
    collapse.weight.data.copy_(torch.cat([out_weights, -out_weights], dim=0))
    collapse.bias.data.copy_(torch.tensor([output_bias, -output_bias], dtype=dtype))
    layers += [collapse, nn.ReLU()]

    # Each identity-pair layer maps [ReLU(y), ReLU(-y)] back to itself:
    # preactivation = [p - n, n - p] = [y, -y]
    # after ReLU = [ReLU(y), ReLU(-y)]
    for _ in range(L - 3):
        ident = nn.Linear(2, 2, bias=True, dtype=dtype)
        ident.weight.data.copy_(torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=dtype))
        ident.bias.data.zero_()
        layers += [ident, nn.ReLU()]

    final = nn.Linear(2, 1, bias=True, dtype=dtype)
    final.weight.data.copy_(torch.tensor([[1.0, -1.0]], dtype=dtype))
    final.bias.data.zero_()
    layers.append(final)

    return nn.Sequential(*layers)


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
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    if len(xs) > 1 and any(left == right for left, right in zip(xs, xs[1:])):
        raise ValueError("Breakpoints must have distinct coordinates.")

    # Give the generic PWL constructor one auxiliary knot on either side.  Its
    # linear extrapolation then has the requested exterior slopes, while the
    # supplied breakpoints remain the knots of the certificate itself.
    xs = [xs[0] - 1.0, *xs, xs[-1] + 1.0]
    ys = [ys[0] - leftmost_slope, *ys, ys[-1] + rightmost_slope]
    network = _make_pwl_relu_network(
        xs, ys, L=2, dtype=torch.get_default_dtype()
    )

    return Certificate(*network.children()).requires_grad_(False)
