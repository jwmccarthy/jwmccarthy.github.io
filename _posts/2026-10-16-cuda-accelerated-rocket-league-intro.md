---
layout: post
title: "CARL: <strong>C</strong>uda-<strong>A</strong>ccelerated <strong>R</strong>ocket <strong>L</strong>eague"
date: 2026-10-16
mathjax: true
---

![Carl from Jimmy Neutron](/images/carl.webp){: .post-hero-image }


This post introduces CARL, a GPU-resident Rocket League simulator built for reinforcement learning. Inspired by the likes of [IsaacLab](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html) and [PureJaxRL](https://chrislu.page/blog/meta-disco/), CARL's parallel environment enables the rapid training of Rocket League agents. Tens of thousands of games can be run simultaneously on a single GPU, which exposes the policy to significantly more experience in less time than serial simulations.

### Interface
CARL exposes the underlying simulation via the standard Gymnasium API. It is planned to add support for other ML libraries like JAX and Tensorflow, but the default wrapper at the moment is written in PyTorch. As with all other Gymnasium-wrapped environments, one need only initialize, reset, and step!
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
#### Action Space
Internally, CARL represents actions with shape $(n_{\mathrm{sim}}, n_{\mathrm{cars}}, 7)$, though users may treat each car as its own "environment", where actions are submitted in the shape $(n_{\mathrm{sim}} * n_{\mathrm{cars}}, 7)$.

Actions by default are continuous, though the user can specify discrete actions via `discrete_actions=True`. In that case, the maximum continuous action is applied.

<details markdown="1">
<summary>Action controls</summary>

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

</details>

Actions are held for every tick over the subsequent `frameskip` ticks.
#### Observation Space
The observation space has shape $(n_{\mathrm{sim}} n_{\mathrm{cars}}, d_{\mathrm{obs}})$, where each `float32` row describes one car's view of the game. Internally, CARL groups observations as $(n_{\mathrm{sim}}, n_{\mathrm{cars}}, d_{\mathrm{obs}})$.

<details markdown="1">
<summary>Observation features</summary>

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

</details>

`CARLObservation` provides convenient named tensor views over these blocks:

<details markdown="1">
<summary>`CARLObservation` views</summary>

```python
observation.ball.position           # (n_envs, 3)
observation.cars.ego.boost          # (n_envs,)
observation.boost_pad_active        # (n_envs, 34)
observation.ego_ball_relative       # (n_envs, 6)
observation.ego_other_relative      # (n_envs, n_cars - 1, 6)
observation.own_goal_relative       # (n_envs, 3)
observation.ego_has_flip_or_jump    # (n_envs,)
```

</details>

By default CARL normalizes observations via the maximum values of arena extent, velocity, etc. though one may disable this via `normalize=False` in the environment constructor.

### Implementation

The game state resides entirely on the GPU in a struct-of-arrays layout.
