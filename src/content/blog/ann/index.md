---
title: "PINNs part I - Neural Networks"
description: "Physics-Informed Neural Networks (PINNs) are a specialized class of deep learning models that embed underlying physical laws directly into their architecture or optimization process. They are primarily designed to serve as highly efficient surrogate models, offering an alternative to computationally expensive traditional numerical solvers."
date: 2026-07-10
authors:
  - adilthami
image: ./assets/PINN_schema.svg
tags:
  - PINNs
  - Deep Learning
  - Inverse modelling
# draft: true
---

*This blog post is directly inspired by a seminar I presented during my time at the Earth and Life Institute at UCLouvain.*

## PINNs are Neural Networks

Historically, an **artificial neural network (ANN)** is described as a computational model loosely inspired by the connectivity of biological neurons. Mathematically, it is more useful to think of it simply as a parametric function:

$$
\hat{\mathbf y} = f(\mathbf x;\boldsymbol\theta),
$$

where $\mathbf x$ is the input, $\hat{\mathbf y}$ is the prediction, and $\boldsymbol\theta$ are the parameters the network learns — its weights and biases:

$$
\boldsymbol\theta = \{\mathbf W, \mathbf b\}
$$

**Deep learning** simply means stacking many of these transformations, called *layers* $L$, one after another, so that simple pieces combine into something that can represent complex patterns. A network with more than one layer is a **deep neural network (DNN)**.

### Two ways to build $f$

- **Linear model.** $\hat{\mathbf y} = \mathbf W^\top\mathbf x + \mathbf b$: the prediction is an affine transformation of the raw input features.
- **Generalized (nonlinear) model.** $\hat{\mathbf y} = \boldsymbol\phi(\mathbf x;\boldsymbol\theta)^\top \mathbf w$: the prediction is a combination of a nonlinear function $\boldsymbol\phi$, and a final linear transformation.

The second option represents how ANNs operate. It learns $\boldsymbol\phi$ automatically, at the same time as it learns the final weights $\mathbf w$, just by gradient descent. Therefore, a network with just one hidden layer, given enough neurons, can approximate essentially any function. Two caveats: saying such a network *exists* doesn't say how big it needs to be, and it says nothing about whether training will actually find it.

### Forward propagation

A network is built from $L$ layers stacked one after another. Let the network have $L$ layers indexed $l=1,\dots,L$, with layer widths $n_0$ (input), $n_1,\dots,n_{L-1}$ (hidden), $n_L$ (output). Each layer takes the previous layer's output, performs a linear transformation, and passes it through a nonlinearity:

$$
\mathbf a^{(0)} := \mathbf x,
$$

$$
\mathbf z^{(l)} = \mathbf W^{(l)}\mathbf a^{(l-1)} + \mathbf b^{(l)}, \qquad
\mathbf a^{(l)} = \phi^{(l)}\left(\mathbf z^{(l)}\right), \qquad l = 1,\dots,L,
$$

Here $\mathbf z^{(l)}$ is the "pre-activation" signal at layer $l$, and $\mathbf a^{(l)}$ (or hidden state) is that same signal after passing through the activation function $\phi^{(l)}$ applied **elementwise**. The final layer's output *is* the prediction:

$$
\hat{\mathbf y} := \mathbf a^{(L)} = f(\mathbf x;\boldsymbol\theta).
$$

So there's really only one rule, applied repeatedly.

### Training by backpropagation

To train the network, we compare its prediction $\hat{\mathbf y}$ to the true value $\mathbf y$ using a loss function $\mathcal L(\hat{\mathbf y},\mathbf y)$, e.g., the mean squared error $\mathcal L = \tfrac12\lVert \hat{\mathbf y}-\mathbf y\rVert^2$. Training means adjusting $\boldsymbol\theta$ to make this loss as small as possible, on average, across the training data.

**Backpropagation** is just a systematic way of using the chain rule of derivation to figure out how much each weight contributed to the error, layer by layer, working backward from the output.

Define an "error signal" $\boldsymbol\delta^{(l)}$ at each layer — how sensitive the loss is to that layer's raw signal $\mathbf z^{(l)}$. It's computed backward, starting at the output:

**At the output layer:**
$$
\boldsymbol\delta^{(L)} = \nabla_{\mathbf a^{(L)}}\mathcal L \;\odot\; \phi'\!\left(\mathbf z^{(L)}\right),
$$

**At each hidden layer, working backward:**
$$
\boldsymbol\delta^{(l)} = \left(\big(\mathbf W^{(l+1)}\big)^{\!\top}\boldsymbol\delta^{(l+1)}\right)\odot \phi'\!\left(\mathbf z^{(l)}\right),
$$

(The symbol $\odot$ means "multiply element by element," not a matrix product — the two things being multiplied are lists of numbers of the same length, one per neuron.)

Once you have $\boldsymbol\delta^{(l)}$, the gradients you actually need for the weight update fall out directly:
$$
\frac{\partial\mathcal L}{\partial \mathbf W^{(l)}} = \boldsymbol\delta^{(l)}\big(\mathbf a^{(l-1)}\big)^{\!\top}, \qquad
\frac{\partial\mathcal L}{\partial \mathbf b^{(l)}} = \boldsymbol\delta^{(l)},
$$

and the update itself is just a small step downhill, with learning rate $\eta$:
$$
\mathbf W^{(l)} \leftarrow \mathbf W^{(l)} - \eta\,\frac{\partial\mathcal L}{\partial \mathbf W^{(l)}}, \qquad
\mathbf b^{(l)} \leftarrow \mathbf b^{(l)} - \eta\,\frac{\partial\mathcal L}{\partial \mathbf b^{(l)}}.
$$

Put together, training is a loop: run the network forward, compute the loss, run backpropagation to get the gradients, take a small step downhill, and repeat — until the loss is small enough.

### References

- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.