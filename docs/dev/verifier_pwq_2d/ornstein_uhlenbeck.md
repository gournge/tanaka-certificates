# Explicit two-dimensional Ornstein--Uhlenbeck PWQ example

This example exercises the exact two-dimensional piecewise-quadratic verifier
without requiring numerical certificate synthesis. All coefficients and
verification margins can be checked directly. The executable version is
[`test_verifier_pwq_2d_ou_example.py`](../../../tests/verifier/test_verifier_pwq_2d_ou_example.py),
and the problem data are defined by `make_piecewise_quadratic_ou_2d_problem` in
[`problems.py`](../../../tanaka_certificates/problems.py).

## Dynamics and generator

Consider the centered isotropic Ornstein--Uhlenbeck process

$$
dX_t=-X_t dt+\frac12 I_2 dW_t,
 X_t=(X_t^1,X_t^2)\in\mathbb R^2.
$$

For a quadratic piece in the verifier's convention,

$$
V_i(x)=x^\top Q_i x+p_i^\top x+c_i,
$$

we have

$$
\nabla V_i(x)=2Q_i x+p_i,

\nabla^2V_i(x)=2Q_i.
$$

Since the drift is $-x$ and the diffusion covariance is
$(I_2/2)(I_2/2)^\top=I_2/4$, the infinitesimal generator is

$$
\begin{aligned}
\mathcal LV_i(x)
&=-x^\top\nabla V_i(x)
  +\frac12\operatorname{tr}\left(\frac14I_2\nabla^2V_i(x)\right)\\
&=-2x^\top Q_i x-p_i^\top x+\frac14\operatorname{tr}(Q_i).
\end{aligned}
$$

## Reach--avoid problem

Let

$$
\begin{aligned}
D   &=[0,1]\times[-1,1],\\
X_0 &=[3/10,2/5]\times[-1/10,1/10],\\
X_u &=[9/10,1]\times[-1,1],\\
X_T &=[0,1/10]\times[-1,1].
\end{aligned}
$$

The certificate thresholds are

$$
\alpha=\frac38,

\beta=\frac35,

\varepsilon=\frac3{20}.
$$

In particular, $\alpha<\beta$. The target is a strip near the stable OU
equilibrium, the initial set is an intermediate rectangle, and the unsafe set
is the strip at the right side of the domain.

## Two-cell certificate

Split the domain at $x_1=1/2$:

$$
K_1=D\cap\{x_1\leq1/2\},

K_2=D\cap\{x_1\geq1/2\}.
$$

Define

$$
V(x)=
\begin{cases}
V_1(x)=x_1-\dfrac14x_1^2,
    &x\in K_1,\\[6pt]
V_2(x)=-\dfrac18x_1^2+\dfrac58x_1+\dfrac5{32},
    &x\in K_2.
\end{cases}
$$

Thus the cell coefficients are

$$
Q_1=\begin{pmatrix}-1/4&0\\0&0\end{pmatrix},

p_1=\begin{pmatrix}1\\0\end{pmatrix},

c_1=0,
$$

and

$$
Q_2=\begin{pmatrix}-1/8&0\\0&0\end{pmatrix},

p_2=\begin{pmatrix}5/8\\0\end{pmatrix},

c_2=\frac5{32}.
$$

Although the certificate is independent of $x_2$, this remains a
two-dimensional stochastic verification problem: the SDE has two noise
coordinates, and the verifier clips and checks two-dimensional cells and
regions.

## 1. Continuity at the interface

At $x_1=1/2$,

$$
V_1(1/2,x_2)
=\frac12-\frac14\left(\frac12\right)^2
=\frac12-\frac1{16}
=\frac7{16}.
$$

For the right piece,

$$
V_2(1/2,x_2)
=-\frac18\left(\frac12\right)^2
 +\frac58\left(\frac12\right)+\frac5{32}
=-\frac1{32}+\frac5{16}+\frac5{32}
=\frac7{16}.
$$

Therefore $V_1=V_2$ on the whole interface and $V$ is continuous.

## 2. Initial and unsafe inequalities

On $K_1$,

$$
\frac{\partial V_1}{\partial x_1}=1-\frac12x_1>0
 (0\leq x_1\leq1/2).
$$

The maximum on $X_0$ is consequently attained at $x_1=2/5$:

$$
\sup_{x\in X_0}V(x)
=V_1(2/5)
=\frac25-\frac14\frac4{25}
=\frac9{25}
<\frac38
=\alpha.
$$

Similarly,

$$
\frac{\partial V_2}{\partial x_1}
=\frac58-\frac14x_1>0
 (1/2\leq x_1\leq1),
$$

so the minimum on $X_u$ is attained at $x_1=9/10$:

$$
\begin{aligned}
\inf_{x\in X_u}V(x)
&=V_2(9/10)\\
&=-\frac18\frac{81}{100}
  +\frac58\frac9{10}+\frac5{32}\\
&=\frac{247}{400}
>\frac35
=\beta.
\end{aligned}
$$

This proves the required separation between the initial and unsafe sets.

## 3. Generator inequality

For the left piece, substitution into the OU generator gives

$$
\mathcal LV_1(x)
=\frac12x_1^2-x_1-\frac1{16}.
$$

Outside the target, the part of $K_1$ that needs checking has
$1/10<x_1\leq1/2$. The derivative of this generator polynomial is
$x_1-1<0$, so its supremum is bounded by its value at $x_1=1/10$:

$$
\mathcal LV_1(x)
\leq
\frac12\frac1{100}-\frac1{10}-\frac1{16}
=-\frac{63}{400}
<-\frac3{20}
=-\varepsilon.
$$

For the right piece,

$$
\mathcal LV_2(x)
=\frac14x_1^2-\frac58x_1-\frac1{32}.
$$

Its derivative is $x_1/2-5/8<0$ on $[1/2,1]$. Therefore its maximum
on $K_2$ occurs at $x_1=1/2$:

$$
\mathcal LV_2(x)
\leq
\frac14\frac14-\frac58\frac12-\frac1{32}
=-\frac9{32}
<-\frac3{20}
=-\varepsilon.
$$

The generator inequality therefore holds everywhere in $D\setminus X_T$,
which is stronger than the verifier's required check only in the sub-$\beta$
basin.

## 4. Interface concavity condition

Let $n=(1,0)^\top$ point from $K_1$ to $K_2$. The two gradients are

$$
\nabla V_1(x)=\begin{pmatrix}1-x_1/2\\0\end{pmatrix},

\nabla V_2(x)=\begin{pmatrix}5/8-x_1/4\\0\end{pmatrix}.
$$

On $x_1=1/2$ their normal derivative jump is

$$
\begin{aligned}
(\nabla V_2-\nabla V_1)^\top n
&=\left(\frac12-\frac34\right)\\
&=-\frac14<0.
\end{aligned}
$$

Thus the interface local-time contribution in the Itô--Tanaka formula is
nonpositive.

## Conclusion

The certificate is continuous, separates the initial and unsafe sets, has
$\mathcal LV\leq-\varepsilon$ outside the target, and has a negative normal
derivative jump at its only interface. Because the domain is compact and both
piecewise gradients are bounded there, the stopped stochastic-integral term is
a true martingale. Hence the certificate satisfies the sufficient conditions
implemented by the exact PWQ verifier.

Run the exact test with

```console
uv run pytest -q tests/verifier/test_verifier_pwq_2d_ou_example.py
```

The plotting script
[`plot_pwq_2d_ou_example.py`](../../../scripts/plot_pwq_2d_ou_example.py) displays
the certificate surface, the reach--avoid regions, sample paths of $X_t$, the
generator, and the corresponding paths of $V(X_t)$. Run it with

```console
uv run python scripts/plot_pwq_2d_ou_example.py
```
