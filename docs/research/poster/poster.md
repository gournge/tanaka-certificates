# Ito-Tanaka certificates: towards scalable SDE certification through compositionality 

## Introduction

Consider the problem of ensuring that a continous system behaves safely under uncertainty - for example, that a humanoid robot does not fall over while walking. One way to ensure safety is to do many experiments. In case of such complex systems, however, it is not feasible. To address this, Neural Continous-Time Supermartingale Certificates [1] give exact probabilistic guarantees of safety for stochastic differential equations (SDEs), using neural networks.

We will consider SDEs of the form
$$
...
$$
over the compact domain K \subset R^d. 

![Poster](poster.png)
*Figure 1: A reach-avoid problem for an SDE described as*


## Problem setup 

Given an initial set X_0, a target set X_T, unsafe set X_u, and alpha < beta, assume we find a function V: K -> R such that

    sup V ... inf V, inf V >= beta

Denoting V(X_{t wedge \tau wedge {\delta D} wedge {V >= beta}}) = V(X_{\tau \wedge \text{stop}}), as a stochastic process, which  the following holds:

> **Theorem**: 
> If V(X_{\tau \wedge \text{stop}}) is a supermartingale, then 
> P(\tau_{X_u} \wedge \tau_{\delta D} < \tau_{X_T}) <= alpha / beta
> which means that the probability of reaching the unsafe set or exiting the certified domain before reaching the target set is bounded by alpha / beta.

## Ito-Tanaka formula and Certificates

In practice, we want to find such a function V using neural networks. To ensure V is a supermartingale, we have to check some conditions relating V, f, g. Checking these conditions is very expensive. It is cheaper to check these 

Expading the ito tanaka:

We notice the terms:


Now, it turns out that if they admit to certain conditions, we can ensure that V is a supermartingale, as illustrated in the following theorem:

> **Theorem**:
> If the Drift (Generator) Condition and the Local-Time Condition hold, (assuming the previous initial/unsafe/target setup) then V(X_{\tau \wedge \text{stop}}) is a supermartingale.

Given the freedom of V being piecewise twice-continously differentiable, we can formule the following:

> **Core idea**: Speed and Compositionality
> 1. checking the conditions is much less expensive for relu networks
> 2. if we have multiple certificates, we can take their minimum and get a new certificate, which is also a supermartingale, because minimum of piecewise C2 functions is also piecewise C2.  

## ReLU region discovery

Briefly describe the algorithm, with notation like in 2026-07-06.tex

## Edge case due to linearity

<edge case blah blah we need curvature.>

General architecture to be piecewise quadratic:

> $V(x) = \sum_{k=1}^m \lambda_k\phi(z_k(x))+c$
> 
> $z(x) = \texttt{NN}(x) = W_L\texttt{ReLU}(\ldots\texttt{ReLU}(W_1x+b_1)\ldots)+b_L$
> 
> $\phi = \texttt{PiecewiseQuadratic1D}$

## Problem of checking pairwise and ICNNs

## Training a certificate

<Example of the comparison plot from 

/home/filipmorawiec/workspace/University/TU Delft/Birmingham/tanaka-certificates-home/second-workspace/tanaka-certificates/output/59efc5b_2026-07-14_00-28-20_verified_radial_ou_certificate

Here also share the training recipe and loss functions, using the pseudocode python notation like in 2026-07-06.tex.
