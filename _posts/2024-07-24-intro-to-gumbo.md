---
layout: post
title: "Cheffing Up RL (pt. 2) - Advantage Estimation"
categories: RL, PyTorch, Optimization
custom_excerpt: "Documenting interesting bits from the development process of Gumbo, my reinforcement learning framework."
---

Someone important once said that early optimization is the enemy of progress, but in some cases it's fun. This post outlines one of those cases as it pertains to advantage estimation in my RL framework!

Generalized Advantage Estimation (GAE) is commonly used in policy-gradient methods to estimate the advantages with lower variance. The form of the GAE estimate is

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{\infty}(\gamma\lambda)^l \delta_{t+l}^V$$

where

$$\delta_{t}^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$

is the TD error for some value function \\(V\\). This can be modeled as a discrete convolution of the TD residuals with a kernel of the \\((\gamma \lambda)^l\\) geometric series.

```python
import torch
import torch.nn.functional as F

def geometric_series(val, n):
    series = torch.ones(n)
    for i in range(1, n):
        series[i] = series[i-1] * val
    return series

def gae_estimate(rewards, values, final_value):
    T = len(rewards)

    # compute TD errors
    next_vals = torch.cat([values[1:], final_value])
    td_errors = (rewards + gamma * next_vals - values).view(1, 1, -1)

    # conv kernel of discount factors
    kernel = geometric_series(lmbda * gamma, T).view(1, 1, -1)

    advantages = F.conv1d(td_errors, kernel, padding=T - 1).view(-1)[-T:]

    return advantages
```

Ordinarily implemented with a for loop, GAE calculation via a convolution provides a decent speedup for reasonably-sized episodes, but it suffers as the array length gets long. 

Fortunately, there's a more efficient alternative which leverages the Fast Fourier Transform (FFT). By the convolution theorem, for functions \\(f\\) and \\(g\\),

$$FT(f*g) = FT(f) \cdot FT(g)$$

which means we can obtain the convolution of \\(f\\) and \\(g\\) as

$$f*g = FT^{-1}(FT(f) \cdot FT(g))$$

The Torchaudio library provides an implementation of this which we can leverage for advantage estimation.

```python
from torchaudio.functional import fftconvolve

def gae_estimate_fft(rewards, values, final_value):
    T = len(rewards)

    # extra dimensions not needed
    next_vals = torch.cat([values[1:], final_value])
    td_errors = rewards + gamma * next_vals - values

    # fftconvolve expects reversed kernel
    kernel = geometric_series(lmbda * gamma, T).flip(0)

    advantages = fftconvolve(td_errors, kernel)[-T:]

    return advantages
```

The run times on various input sizes are shown below. Warm up runs were performed for GPU methods to obtain better runtime estimates.
![GAE Calculation Runtimes Compared](/assets/gae_runtime.png)
We can see orders of magnitude improvements in runtime with the FFT convolution, which for more complex environments and longer experience episodes will save us a lot of compute time. Pretty cool!
