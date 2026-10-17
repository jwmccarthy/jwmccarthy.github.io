---
layout: post
title: "CARL: <strong>C</strong>uda-<strong>A</strong>ccelerated <strong>R</strong>ocket <strong>L</strong>eague"
date: 2026-10-16
mathjax: true
---

![Carl from Jimmy Neutron](/images/carl.webp){: .post-hero-image }


This post introduces CARL, a GPU-resident Rocket League simulator built for reinforcement learning. Inspired by the likes of [IsaacLab](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html) and [PureJaxRL](https://chrislu.page/blog/meta-disco/), CARL's parallel environment enables the rapid training of Rocket League agents. Tens of thousands of games can be run simultaneously on a single GPU, which exposes the policy to significantly more experience in less time than serial simulations.

### Interface
CARL exposes the underlying simulation via the standard Gymnasium API.
```python
import torch
from carl.gymnasium import CARLTorchVectorEnv

env = CARLTorchVectorEnv(n_sim=8192)

observation = env.reset()

actions = torch.zeros(
    env.action_space.shape,
    dtype=torch.float32,
    device=env.device
)

observation, reward, terminated, truncated, info = env.step(actions)
```
Internally, CARL represents actions with shape $(n_{\mathrm{sim}}, n_{\mathrm{cars}}, 7)$, though users may treat each car as its own "environment" such that the action space is of shape $(n_{\mathrm{sim}} * n_{\mathrm{cars}}, 7)$.

Actions by default are continuous; set `discrete_actions=True` to use discrete controls instead.

| Action | Continuous shape | Continuous range | Discrete shape | Discrete range |
|---|---|---|---|---|
| Horizontal | $()$ | $[-1, 1]$ | $()$ | $\{0, 1, 2\}$ |
| Vertical | $()$ | $[-1, 1]$ | $()$ | $\{0, 1, 2\}$ |
| Throttle | $()$ | $[-1, 1]$ | $()$ | $\{0, 1, 2\}$ |
| Powerslide | $()$ | $[0, 1]$ | $()$ | $\{0, 1\}$ |
| Boost | $()$ | $[0, 1]$ | $()$ | $\{0, 1\}$ |
| Air roll | $()$ | $[-1, 1]$ | $()$ | $\{0, 1, 2\}$ |
| Jump | $()$ | $[0, 1]$ | $()$ | $\{0, 1\}$ |
