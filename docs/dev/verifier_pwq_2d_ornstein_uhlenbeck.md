# Two-dimensional Ornstein–Uhlenbeck PWQ verifier test

This test is intended to exercise reach-avoid verification for a continuous
piecewise-quadratic (PWQ) certificate in two dimensions. The stochastic process
is the isotropic Ornstein–Uhlenbeck SDE
$dX_t=-X_t\,dt+0.5I_2\,dW_t$, where $X_t\in\mathbb R^2$ and $W_t$ is a
two-dimensional Wiener process.
The drift pulls trajectories towards the origin, while the relatively small
diffusion makes the inward mean-reverting behaviour clearly visible without
removing stochastic variation.

## Reach-avoid geometry

The verification domain is the rectangle $D=[-1,1.25]\times[-1.25,0.75]$.

The initial set is a square of half-width $0.1$, centred at $(1,-1)$:
$X_0=[0.9,1.1]\times[-1.1,-0.9]$.

The target is the square of half-width $0.1$ centred at the stable equilibrium:
$X_T=[-0.1,0.1]^2$.

There are two unsafe squares, each with half-width $0.2$. They are centred at
$(0,-1)$ and $(1,0)$: $X_U=([-0.2,0.2]\times[-1.2,-0.8])\cup([0.8,1.2]\times[-0.2,0.2])$.

Their placement makes the geometry non-radial. Although the OU drift points
directly towards the target, a trajectory following a strongly axis-aligned
route can encounter an unsafe set. A certificate must therefore distinguish
safe inward motion from motion through either obstacle; a simple function of
the distance to the origin need not capture this geometry.

## Certificate conditions

The test compares the strict level $\alpha=0.5$ with the more permissive level
$\alpha=1.5$. Both cases use $\beta=2$ and $\epsilon=0.1$. A valid certificate
$V$ must establish
$\sup_{x\in X_0}V(x)\leq\alpha$ and
$\inf_{x\in X_U}V(x)\geq\beta$.

On every smooth quadratic piece, throughout the sublevel basin outside the
target, it must also satisfy $\mathcal LV(x)\leq-\epsilon$ for
$x\in D\cap\{V\leq\beta\}\setminus X_T$.

For a quadratic piece $V_i(x)=x^\top Q_i x+q_i^\top x+c_i$, the OU generator
is $\mathcal LV_i(x)=(2Q_i x+q_i)^\top(-x)+0.25\,\operatorname{tr}(Q_i)$.

The final term is the Itô correction: in general it is
$\tfrac12\operatorname{tr}(GG^\top\nabla^2V_i)$, and here $G=0.5I_2$ and
$\nabla^2V_i=2Q_i$.

Across interfaces between quadratic pieces, the formulas should agree in value
so that $V$ is continuous. The jump in the normal derivative must additionally
have the sign required to prevent a positive local-time contribution in the
generalized Itô–Tanaka formula. These interface checks are the genuinely
piecewise part of the test and cannot be replaced by checking the smooth
generator inside each region alone.

Each test trains the PWQ certificate directly and then invokes the
exact verifier with an OU generator implementation local to the test. The
$\alpha=1.5$ case is the intended feasible case, but is currently marked as an
expected failure because the baseline optimizer does not yet synthesize a
valid certificate. The $\alpha=0.5$ case is expected not to verify. Keeping
both cases in the same two-test file makes the intended training regression
explicit without injecting a synthetic generator bound.

## Visualization

![Two-dimensional OU reach-avoid geometry](img/verifier_pwq_2d_ornstein_uhlenbeck.png)

The trajectories in the figure are simulations for intuition only. Verification
must rely on certified bounds over whole regions, not on the sampled paths.
