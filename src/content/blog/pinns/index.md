---
title: "PINNs part II - Physics Informed Neural Networks"
description: "Physics-Informed Neural Networks (PINNs) are a specialized class of deep learning models that embed underlying physical laws directly into their architecture or optimization process. They are primarily designed to serve as highly efficient surrogate models, offering an alternative to computationally expensive traditional numerical solvers."
date: 2026-07-11
authors:
  - adilthami
image: ./assets/PINN_schema.svg
tags:
  - PINNs
  - Deep Learning
  - Inverse modelling
draft: true
---

*This blog post is directly inspired by a seminar I presented during my time at the Earth and Life Institute at UCLouvain.*

## From data-driven to physics-informed learning

An ordinary network only ever sees pairs of input and true output. Nothing stops it from producing predictions that quietly break the laws of physics — and getting it to behave well usually takes a lot of data.

**Physics-informed neural networks (PINNs)** fix this by adding a new ingredient to the loss: instead of only checking "does the prediction match the data?", we also check "does the prediction obey the governing equation?" — everywhere, even where we have no data at all.

### The general idea

Say we know the physical system obeys some differential equation — schematically,
$$
\mathcal N[u] = f,
$$
where $u$ is the true (usually unknown) solution and $\mathcal N$ is a differential operator built from derivatives (think: Newton's second law, the heat equation, Navier–Stokes, Schrödinger's equation — whatever governs the system). There are usually also initial or boundary conditions the solution must satisfy.

We replace $u$ with a neural network $u_{\boldsymbol\theta}$, and ask: how far is $\mathcal N[u_{\boldsymbol\theta}]$ from $f$? This gap is the **residual**:
$$
r_{\boldsymbol\theta} := \mathcal N[u_{\boldsymbol\theta}] - f.
$$

If $u_{\boldsymbol\theta}$ were the exact solution, this residual would be zero everywhere. The trick that makes this computable is **automatic differentiation**: since the network is built out of simple, differentiable pieces, we can compute its exact derivatives with respect to its own inputs (not just with respect to its weights, as in ordinary backpropagation) at essentially any point we like — no approximation needed.

The loss becomes a mix of terms:
$$
\mathcal L = \lambda_{\text{phys}}\,\mathcal L_{\text{phys}} + \lambda_{\text{ic/bc}}\,\mathcal L_{\text{ic/bc}} + \lambda_{\text{data}}\,\mathcal L_{\text{data}},
$$
where $\mathcal L_{\text{phys}}$ is the average squared residual, sampled at many points across the domain; $\mathcal L_{\text{ic/bc}}$ penalizes violating the initial or boundary conditions; and $\mathcal L_{\text{data}}$ is the usual "match the data" term, used wherever real measurements exist. The $\lambda$'s are just weights balancing how much each term matters.

This is the key benefit: the physics term can be checked at *any* point, without needing a labelled answer there — so the network learns from the equation itself, not only from scarce data.

---

## Study case: the damped harmonic oscillator

A mass on a spring, with some friction, follows a simple equation:
$$
m\,\ddot x(t) + c\,\dot x(t) + k\,x(t) = 0,
$$
with a known starting position $x(0)=x_0$ and starting velocity $\dot x(0)=v_0$. This has a textbook closed-form solution (a decaying oscillation), which makes it a good sanity check for a PINN before trying something harder.

**Setup.** A tiny network $x_{\boldsymbol\theta}$ takes time $t$ as input and outputs a guess for the position, $\hat x(t)$.

**Derivatives, for free.** Automatic differentiation gives us the network's velocity $\dot{\hat x}(t)$ and acceleration $\ddot{\hat x}(t)$ exactly, at any $t$ we choose — no finite-difference approximation needed.

**Residual.** Plug these into the equation of motion:
$$
r_{\boldsymbol\theta}(t) = m\,\ddot{\hat x}(t) + c\,\dot{\hat x}(t) + k\,\hat x(t).
$$
If the network has learned the right motion, this is zero.

**Loss.** Pick a bunch of time points $t_1,\dots,t_N$ across the interval of interest, and penalize the residual there, plus penalize missing the known starting position and velocity:
$$
\mathcal L(\boldsymbol\theta) = \frac{1}{N}\sum_{i=1}^{N} r_{\boldsymbol\theta}(t_i)^2
\;+\; \lambda\Big[\big(\hat x(0)-x_0\big)^2 + \big(\dot{\hat x}(0)-v_0\big)^2\Big].
$$

Train as usual: forward pass, differentiate to get velocity and acceleration, compute the loss, backpropagate, update the weights, repeat. If you also don't know $m$, $c$, or $k$ and want to recover them from a few noisy measurements, you simply let the optimizer learn those too, alongside $\boldsymbol\theta$ — this is the "inverse" problem, and it's one of the main reasons PINNs are useful in practice.

---

## Applications, advantages, and limitations

**Where PINNs get used:** fluid dynamics (flow reconstruction, heat transfer, groundwater flow), solid mechanics (stress and strain), quantum mechanics (the Schrödinger equation), electromagnetics (Maxwell's equations) — anywhere a governing equation is known but data is scarce.

**Advantages**

- The physics acts as a built-in constraint, instead of relying only on data to figure out the right behaviour.
- Because the physics can be checked anywhere, PINNs typically need far less labelled data, and cope better with incomplete datasets.
- Predictions tend to stay physically sensible, even between data points.

**Limitations**

- Training is expensive: every step needs derivatives of derivatives (differentiating the network to build the physics loss, then differentiating *that* to update the weights), which is heavier than ordinary training and benefits a lot from GPUs.
- You need to actually know the governing equation — get it wrong, and the network will confidently learn the wrong physics.
- Balancing the different pieces of the loss (physics vs. data vs. boundary conditions) is fiddly, and training can get stuck on equations that are stiff or have very different scales at play.

One small caveat on "generalization": PINNs really do generalize better than an unconstrained network trained on the same data, because the equation holds everywhere, not just near the data. But they're not magic — they can still struggle to represent very fine, high-frequency details, and can still fail badly far outside the region they were trained on (Wang, Teng & Perdikaris, 2021; Krishnapriyan et al., 2021).

---

## Summary

A PINN is an ordinary neural network trained the ordinary way, with one twist: part of its loss checks whether it obeys a known physical equation, using exact derivatives computed via automatic differentiation.

| | Ordinary network | PINN |
|---|---|---|
| Loss | match the data | match the data **and** obey the equation |
| Needs labels everywhere? | Yes | No — physics can be checked anywhere, labels only where available |
| Extra ingredient | — | derivatives of the network's own output, via automatic differentiation |
| Main risk | doesn't generalize past its data | wrong physics, or a fussy, hard-to-balance loss |

---

### References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
- Lagaris, I. E., Likas, A., & Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Transactions on Neural Networks*, 9(5), 987–1000.
- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
- Raissi, M., Yazdani, A., & Karniadakis, G. E. (2020). Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations. *Science*, 367(6481), 1026–1030.
- Cybenko, G. (1989). Approximation by superpositions of a sigmoidal function. *Mathematics of Control, Signals and Systems*, 2(4), 303–314.
- Hornik, K. (1991). Approximation capabilities of multilayer feedforward networks. *Neural Networks*, 4(2), 251–257.
- Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*, 43(5), A3055–A3081.
- Krishnapriyan, A., et al. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS*.