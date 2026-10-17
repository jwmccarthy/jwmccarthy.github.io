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
Internally, CARL represents actions with shape $(n_{\mathrm{sim}}, n_{\mathrm{cars}}, 7)$, though users may treat each car as its own "environment", where actions are submitted in the shape $(n_{\mathrm{sim}} * n_{\mathrm{cars}}, 7)$.

Actions by default are continuous, though the user can specify discrete actions via `discrete_actions=True`. In that case, the maximum continuous action is applied.

| Action | Continuous | Discrete | Effect |
|---|---|---|---|
| Horizontal | $[-1, 1]$ | $0, 1, 2$ | Steer on the ground and yaw in the air |
| Vertical | $[-1, 1]$ | $0, 1, 2$ | Pitch the car up or down |
| Throttle | $[-1, 1]$ | $0, 1, 2$ | Drive forward or reverse |
| Powerslide | $[0, 1]$ | $0, 1$ | Hold to drift and turn sharply |
| Boost | $[0, 1]$ | $0, 1$ | Hold to consume boost for acceleration |
| Air roll | $[-1, 1]$ | $0, 1, 2$ | Roll the car while airborne |
| Jump | $[0, 1]$ | $0, 1$ | Jump from the ground or dodge in the air |
{: .action-table}

The observation space has shape $(n_{\mathrm{sim}} n_{\mathrm{cars}}, d_{\mathrm{obs}})$, where each `float32` row describes one car's view of the game.

| Block | Contents |
|---|---|
| Ball | Position, linear velocity, and angular velocity |
| Cars | Position, velocity, angular velocity, orientation, boost, and state flags |
| Boost pads | Active state and distance to the observing car |
| Self to ball | Relative position and velocity |
| Self to other cars | Relative position and velocity for every other car |
| Goals | Ball displacement to each goal, relative to the observing team |
| Ego | Flip and jump availability, plus remaining dodge-window time |
{: .action-table .observation-table}
