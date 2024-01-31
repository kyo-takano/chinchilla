# Math & Computation

## Formulation

The scaling law model describes how an increase in parameters and data leads to _decelerating_ performance growth.

### Loss Predictor

$$L(N, D | A, B, \alpha, \beta) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

This equation is the parametric formulation of the neural scaling law, predicting the loss of an $N$-parameter model trained on $D$ data points.
  - $L$ is the loss,
  - $N$ is the number of parameters _per prediction_ (e.g., Mixture-of-Expert model uses only a subset of parameter groups for a single forward pass),
  - $D$ is the number of data samples,
  - $E$, $A$, $B$, $\alpha$, and $\beta$ are the parameters to be estimated.

**Composition**:
  1. $E$ represents the minimum loss possible with infinite parameters and data: i.e., the intrinsic entropy/noise of the target data.
  2. $A/{N^\alpha}$ captures the additional loss you could not help even with infinite data, constrained by model size/capacity.
  3. $B/{D^\beta}$ captures the additional loss you could not help even with infinite model size, constrained by training data

### Compute-Optimal Allocation

$$G = \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha + \beta}}$$

$$N_{\text{opt}} = G \left( \frac{C}{6} \right)^{\beta/(\alpha + \beta)}$$

$$D_{\text{opt}} = \frac{1}{G} \left( \frac{C}{6} \right)^{\alpha/(\alpha + \beta)}$$

These equations determine the optimal number of parameters ($N_{\text{opt}}$) and the optimal number of data samples ($D_{\text{opt}}$) for a given computational budget $C$.
$G$, $\beta/(\alpha + \beta)$, and $\alpha/(\alpha + \beta)$ are derived from the scaling law parameters.
$G$ is an instrumental term to balance the allocation to $N$ and $D$; $G<1$ gives more data samples than model parameters, assuming the equivalence of $\alpha$ and $\beta$.
$\beta/(\alpha + \beta)$ and $\alpha/(\alpha + \beta)$ represent the relative contribution of power-scaling $N$ and $D$ respectively, balancing their absolute values with $G$

### Optimal Amount of Data Given $N$

$$D = G^{-\left(1 + \frac{\alpha/(\alpha + \beta)}{\beta/(\alpha + \beta)}\right)} N^{\frac{\alpha/(\alpha + \beta)}{\beta/(\alpha + \beta)}}$$

This formula derives the optimal amount of data $D$ for a given number of parameters $N$, based on the values of $G$, $\beta/(\alpha + \beta)$, and $\alpha/(\alpha + \beta)$.

## Numerical Stability

### Logarithmic Transformations

To avoid underflow/overflow in large-scale computations, we transform some terms into logarithmic space.

$$A/N^\alpha \rightarrow \exp(\log(A) - \alpha \log(N))$$

### Clipping Extreme Values

$$\log(\text{clip}(A, \text{single.tiny}, \text{None}))$$

$$\exp(\text{clip}(A, \text{None}, \log(\text{single.max})))$$

Extreme parameter values are clipped with single-precision limits to maintain stability during optimization.

### Double Precision Arithmetic

Computations are performed in high precision (128-bit and 64-bit) to enhance the expression range before $\log$ and $\exp$.
