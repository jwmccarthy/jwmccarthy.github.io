---
layout: post
title: "CARL: <strong>C</strong>uda-<strong>A</strong>ccelerated <strong>R</strong>ocket <strong>L</strong>eague"
date: 2026-10-16
mathjax: true
---

![Carl from Jimmy Neutron](/images/carl.webp){: .post-hero-image }


This post introduces CARL, a GPU-resident Rocket League simulator built for reinforcement learning. Inspired by the likes of [IsaacLab](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html) and [PureJaxRL](https://chrislu.page/blog/meta-disco/), CARL's parallel environment enables the rapid training of Rocket League agents. Tens of thousands of games can be run simultaneously on a single GPU, which exposes the policy to significantly more experience in less time than serial simulations. Credit goes to [RocketSim](https://github.com/ZealanL/RocketSim) and the [Bullet](https://github.com/bulletphysics/bullet3) physics engine for the ported logic.

## Interface
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
### Action Space
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
### Observation Space
The observation space has shape $(n_{\mathrm{sim}} n_{\mathrm{cars}}, d_{\mathrm{obs}})$, where each `float32` row describes one car's view of the game. Internally, CARL groups observations as $(n_{\mathrm{sim}}, n_{\mathrm{cars}}, d_{\mathrm{obs}})$. `CARLObservation` provides convenient named tensor views over these contiguous blocks. 

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

<details markdown="1">
<summary><code>CARLObservation</code> views</summary>

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

## Implementation

### Game State
The game state resides entirely on the GPU in a struct-of-arrays layout. During simulation, the kernels index into these arrays to perform data-dependent computations or modify state information.

<figure class="post-figure">
  <img src="{{ '/images/aos-soa-memory-layout.png' | relative_url }}" alt="Comparison of array-of-structures and struct-of-arrays memory layouts">
  <figcaption>Array-of-structures and struct-of-arrays memory layouts. <a href="#reference-zhao-et-al-2022">[1]</a></figcaption>
</figure>

SoA is the optimal approach for data storage in GPU applications due to the enforcement of coalesced fetch instructions for contiguous elements. If we were instead to store arrays of structs, each associated with a single game's state, the GPU would be forced to perform strided accesses, leading to more memory transactions. We'd also be pulling in full structs worth of data we may not need at the time.

### Arena Geometry
The arena mesh is loaded into global memory upon environment initialization, and triangles are accessed via CUDA's cache-optimized read-only load intrinsics. The arena is immediately divided into a 3D broad phase grid which will to reduce the number of computation-heavy triangle tests during collision resolution.

CARL leverages the prefix sum construction described in NVIDIA's [Broad-Phase Collision Detection with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda). The prefix sum converts each broad phase cell's triangle occupancy into offsets/counts, allowing the car-arena and ball-arena collision kernels to determine where a cell's triangle list begins and how many triangles it must test.

<figure class="post-figure post-figure-compact">
  <img src="{{ '/images/gpu-gems-spatial-subdivision.jpg' | relative_url }}" alt="A bounding sphere overlapping cells in a broad-phase grid">
  <figcaption>A bounding sphere overlapping broad-phase grid cells. <a href="#reference-le-grand-2007">[2]</a></figcaption>
</figure>

It is quick to check both sphere and axis-aligned bounding box (AABB) occupancy, so we can quickly cull most of the triangles in the arena before checking against the car and ball.

### Simulation Stages
CARL organizes its simulation step into a sequence of kernel calls. This structure was found to be more efficient than a unified kernel, as each step benefited more from optimized structures than was lost in kernel launch overhead.

The stages are broadly as follows:

#### 1. Broad Phase & Suspension
Uniform grid lookup finds each car's occupied cell, then a prefix scan packs its candidate triangles. Suspension raycasts are performed against the arena, ball, and other cars. The latter two are helpful for determining flip resets. Unlike RocketSim's [bounding volume hierarchy (BVH)](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) broad phase, CARL uses this fixed arena grid and scan-packed ranges to make triangle work per car more "regular" on the GPU.

#### 2. Controls
Actions are decoded into controls and their effects are added to the impulse accumulators.

#### 3. Narrow Phase
Unlike RocketSim, CARL does not use [GJK](https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm)/[EPA](https://en.wikipedia.org/wiki/Expanding_polytope_algorithm). Instead, it implements the [Separating Axis Theorem (SAT)](https://en.wikipedia.org/wiki/Hyperplane_separation_theorem#Separating_axis_theorem) to identify car-arena and car-car collisions without iterative support mapping. This leads to slightly different contact manifolds over time relative to RocketSim, but the feel of the game remains the same and short-range divergence is minimal, allowing RL agents to adjust as they play.

Sphere-OBB and sphere-triangle closest point tests generate car-ball and ball-arena contacts. Clipping via [Sutherland-Hodgman](https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm) constructs the car contact manifolds. These manifolds are persistent and updated each tick depending on the generation of new contact points. A maximum of 4 points is allowed per manifold in line with RocketSim. Again, this leads to minor divergences that add up over time given the same controls, but for the purposes of RL this is more than sufficient to transfer learning from simulation to the game itself.

#### 4. Solve
CARL uses a warm-started sequential-impulse solver, a clamped form of the [Gauss-Seidel method](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method). Cached normal impulses seed each car contact manifold, then a separate split-impulse pass corrects penetration without changing the reported velocity. The solver iterates nonnegative normal impulses followed by friction impulses clamped by the accumulated normal impulse. The resulting linear and angular impulses are added to each body's accumulators. Rather than invoking RocketSim/Bullet's serial world-step pipeline, CARL executes these bounded constraint iterations in GPU kernels across its batched simulations.

#### 5. Integration
[Semi-implicit Euler integration](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method) applies the accumulated impulses to the cars and ball, while quaternion updates advance car orientation. CARL also applies Rocket League-specific adjustments: ball drag and gravity, linear and angular speed caps, and an empirical extra ball-velocity impulse for each new car-ball hit. This hit impulse is capped and scaled by relative speed, with its direction adjusted by the car's orientation; per-tick hit tracking prevents it from being applied repeatedly for the same contact.

## References

<ol>
  <li id="reference-zhao-et-al-2022">Zhao, W.-L., Wang, W., and Wang, Q. (2022). <a href="https://doi.org/10.1007/s11227-021-04153-0"><em>Optimization of cosmological N-body simulation with FMM-PM on SIMT accelerators</em></a>. <em>The Journal of Supercomputing</em>, 78.</li>
  <li id="reference-le-grand-2007">Le Grand, S. (2007). <a href="https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda"><em>Broad-Phase Collision Detection with CUDA</em></a>. In <em>GPU Gems 3</em>.</li>
  <li>NVIDIA. (n.d.). <a href="https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html"><em>Isaac Lab</em></a>.</li>
  <li>Lu, C. (n.d.). <a href="https://chrislu.page/blog/meta-disco/"><em>Meta-Discoveries from PureJaxRL</em></a>.</li>
  <li>ZealanL. (n.d.). <a href="https://github.com/ZealanL/RocketSim"><em>RocketSim</em></a> [Source code]. GitHub.</li>
  <li>Coumans, E. (n.d.). <a href="https://github.com/bulletphysics/bullet3"><em>Bullet Physics SDK</em></a> [Source code]. GitHub.</li>
  <li>Wikipedia contributors. (n.d.). <a href="https://en.wikipedia.org/wiki/Hyperplane_separation_theorem#Separating_axis_theorem"><em>Separating axis theorem</em></a>.</li>
  <li>Wikipedia contributors. (n.d.). <a href="https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm"><em>Sutherland-Hodgman algorithm</em></a>.</li>
</ol>
