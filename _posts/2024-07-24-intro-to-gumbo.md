---
layout: post
title: "Working Title (pt.1)"
categories: RL, PyTorch
custom_excerpt: "Assorted topics related to the development of my Reinforcement Learning framework"
---

### What Am I Even Doing?

Hello! Thanks for stopping by. Apologies in advance for the tonal and thematic scratchwork that is my writing. I don't expect it'll ever get much better, but who's to say what better is anyway?

This is my first post but it finds us in the middle of the action. I'm currently working on building a reinforcement learning framework from scratch using PyTorch as the basis for all the components. RL caught my eye with the recent influx of machine learning bots in Rocket League. It seemed like a fun project to get up to speed with the latest methods in reinforcement learning (I am an optimist). Rather than implement various algorithms in existing frameworks, I wanted to create my own. This post will outline a few features of the framework as it exists today.

Proximal Policy Optimization (PPO) is among the current SOTA RL algorithms, and it was used to train the [first generation of ML Rocket League agents](https://github.com/Rolv-Arild/Necto). Many of the design decisions cater to the PPO learning procedure as it's the first I've implemented. If you see any glaring issues with the current structure as it pertains to other algorithms, I know they're in there somewhere, and one day the tech debt gods will smite me.

---

### Experience Data

None of the frameworks I investigated prior to building my own handled the collection and processing of training data in a way that I liked. A common solution was lightweight dataclasses, which are simple but lack functionality. Others were closer to what I wanted. [TorchRL](https://pytorch.org/rl/stable/index.html), for instance, uses what they call a TensorDict, which is a Python dictionary that allows for certain tensor operations.

My solution was to create a custom class that acted as a hybrid between these two methods. This [``TensorDataset``](https://github.com/jwmccarthy/Gumbo/blob/main/gumbo/data/datasets.py) allows for dot attribute access & setting, tensor operations & indexing, dataset unions, and device allocation. The accompanying ``Subset`` serves as a way to partition the broader data into slices that can represent individual episodes while still referencing a common underlying memory. It also stores auxiliary information, such as the final observation following episode truncation, which is important for bootstrapping.

The [``EpisodicBuffer``](https://github.com/jwmccarthy/Gumbo/blob/main/gumbo/data/buffers.py) class serves as the wrapper for managing ``TensorDataset`` instances as it pertains to experience collection. When an episode terminates, a ``Subset`` instance is created that can be used to reference data associated with said episode. Operations (such as advantage estimation) can then be performed in an episode-wise fashion.

---

### Advantage Estimation

This section started small but lead me to an improvement that I'll be implementing in my framework.

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
