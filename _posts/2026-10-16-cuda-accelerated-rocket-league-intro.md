---
layout: post
title: "CARL: <strong>C</strong>uda-<strong>A</strong>ccelerated <strong>R</strong>ocket <strong>L</strong>eague"
date: 2026-10-16
mathjax: true
---

![Carl from Jimmy Neutron](/images/carl.webp){: .post-hero-image }


This post introduces CARL, a GPU-resident Rocket League simulator built for reinforcement learning. Inspired by the likes of [IsaacLab](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html) <a href="#reference-isaac-lab">[3]</a> and [PureJaxRL](https://chrislu.page/blog/meta-disco/) <a href="#reference-purejaxrl">[4]</a>, CARL's parallel environment enables the rapid training of Rocket League agents. Tens of thousands of games can be run simultaneously on a single GPU, which exposes the policy to significantly more experience in less time than serial simulations. Credit goes to [RocketSim](https://github.com/ZealanL/RocketSim) <a href="#reference-rocketsim">[5]</a> and the [Bullet](https://github.com/bulletphysics/bullet3) physics engine <a href="#reference-bullet">[6]</a> for the ported logic.

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
The observation space has shape $(n_{\mathrm{sim}} * n_{\mathrm{cars}}, d_{\mathrm{obs}})$, where each `float32` row describes one car's view of the game. Internally, CARL groups observations as $(n_{\mathrm{sim}}, n_{\mathrm{cars}}, d_{\mathrm{obs}})$. `CARLObservation` provides convenient named tensor views over these contiguous blocks. 

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

CARL leverages the prefix sum construction described in NVIDIA's [Broad-Phase Collision Detection with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda) <a href="#reference-le-grand-2007">[2]</a>. The prefix sum converts each broad phase cell's triangle occupancy into offsets/counts, allowing the car-arena and ball-arena collision kernels to determine where a cell's triangle list begins and how many triangles it must test.

<figure class="post-figure post-figure-compact post-figure-grid">
  <img src="{{ '/images/gpu-gems-cell-id-array.jpg' | relative_url }}" alt="Broad-phase cell ID array after construction">
  <figcaption>Initial contents of the broad-phase cell ID array. <a href="#reference-le-grand-2007">[2]</a></figcaption>
</figure>

It is quick to check both sphere and axis-aligned bounding box (AABB) occupancy, so we can quickly cull most of the triangles in the arena before checking against the car and ball.

### Simulation Stages
CARL organizes its simulation step into a sequence of kernel calls. This structure was found to be more efficient than a unified kernel, as each step benefited more from optimized structures than was lost in kernel launch overhead.

The stages are broadly as follows:

#### 1. Broad Phase & Suspension
Uniform grid lookup finds each car's occupied cell, then a prefix scan packs its candidate triangles. Suspension raycasts are performed against the arena, ball, and other cars. The latter two are helpful for determining flip resets. Unlike [RocketSim](https://github.com/ZealanL/RocketSim) <a href="#reference-rocketsim">[5]</a>'s [bounding volume hierarchy (BVH)](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) <a href="#reference-gottschalk-et-al-1996">[7]</a> broad phase, CARL uses this fixed arena grid and scan-packed ranges to make triangle work per car more "regular" on the GPU.

#### 2. Controls
Actions are decoded into controls and their effects are added to the impulse accumulators.

#### 3. Narrow Phase
Unlike RocketSim, CARL does not use [GJK](https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm)/[EPA](https://en.wikipedia.org/wiki/Expanding_polytope_algorithm) <a href="#reference-van-den-bergen-2004">[11]</a>. Instead, it implements the [Separating Axis Theorem (SAT)](https://en.wikipedia.org/wiki/Hyperplane_separation_theorem#Separating_axis_theorem) <a href="#reference-van-den-bergen-2004">[11]</a> to identify car-arena and car-car collisions without iterative support mapping. This leads to slightly different contact manifolds over time relative to RocketSim, but the feel of the game remains the same and short-range divergence is minimal, allowing RL agents to adjust as they play.

<figure class="post-figure post-figure-half post-figure-sat">
  <img src="{{ '/images/separating-axis-theorem.webp' | relative_url }}" alt="Separating Axis Theorem overlap test">
  <figcaption>Separating Axis Theorem OBB overlap test. <a href="#reference-tayyub-khan-2019">[9]</a></figcaption>
</figure>

Sphere-OBB and sphere-triangle closest point tests generate car-ball and ball-arena contacts. Clipping via [Sutherland-Hodgman](https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm) <a href="#reference-sutherland-hodgman">[8]</a> constructs the car contact manifolds. 

<figure class="post-figure post-figure-half">
  <img src="{{ '/images/sutherland-hodgman-clipping.svg' | relative_url }}" alt="Steps of Sutherland-Hodgman polygon clipping">
  <figcaption>Sutherland-Hodgman polygon clipping. <a href="#reference-sutherland-hodgman">[8]</a> Illustration by <a href="https://commons.wikimedia.org/wiki/File:Sutherland-Hodgman_clipping_sample.svg">Wojciech mula</a>, public domain.</figcaption>
</figure>

These manifolds are persistent and updated each tick depending on the generation of new contact points. A maximum of 4 points is allowed per manifold in line with RocketSim. Again, this leads to minor divergences that add up over time given the same controls, but for the purposes of RL this is more than sufficient to transfer learning from simulation to the game itself.

#### 4. Solve
CARL uses a warm-started sequential-impulse solver, a clamped form of the [Gauss-Seidel method](https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method) <a href="#reference-varga-2000">[12]</a>. Cached normal impulses seed each car contact manifold, then a separate split-impulse pass corrects penetration without changing the reported velocity. The solver iterates nonnegative normal impulses followed by friction impulses clamped by the accumulated normal impulse. The resulting linear and angular impulses are added to each body's accumulators. Rather than invoking RocketSim/Bullet's serial world-step pipeline, CARL executes these bounded constraint iterations in GPU kernels across its batched simulations.

#### 5. Integration
[Semi-implicit Euler integration](https://en.wikipedia.org/wiki/Semi-implicit_Euler_method) <a href="#reference-baraff-1997">[13]</a> applies the accumulated impulses to the cars and ball, while quaternion updates advance car orientation. CARL also applies Rocket League-specific adjustments such as ball drag and gravity, linear and angular speed caps, and an extra ball velocity impulse for each new car-ball hit. This hit impulse is capped and scaled by relative speed, with its direction adjusted by the car's orientation.

### I/O
Below the Gymnasium wrapper, `carl.Env` is a CUDA-only [DLPack](https://dmlc.github.io/dlpack/latest/) interface. <a href="#reference-dlpack">[10]</a> Inputs must be contiguous tensors on the device. This way, no host round trip is required. 

The native batch is organized by match rather than actor. All cars in a simulation share one physics world, score, reward event, and termination/truncation flags. The PyTorch wrapper flattens the leading two dimensions so that each car appears as a separate environment.

`step()` first packs the transition state and terminal observations, then automatically resets completed matches to shuffled kickoffs and produces the next observation buffers. Returned DLPack tensors are non-owning views over CARL's reusable GPU buffers, so callers that need to retain a result across a subsequent `step()` must clone or copy it. The Gymnasium PyTorch wrapper does this by default and expands match-level rewards and terminations back to each participating car.

## References

<ol>
  <li id="reference-zhao-et-al-2022">Zhao, W.-L., Wang, W., and Wang, Q. (2022). <a href="https://doi.org/10.1007/s11227-021-04153-0"><em>Optimization of cosmological N-body simulation with FMM-PM on SIMT accelerators</em></a>. <em>The Journal of Supercomputing</em>, 78(5), 7186-7205.</li>
  <li id="reference-le-grand-2007">Le Grand, S. (2007). <a href="https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda"><em>Broad-phase collision detection with CUDA</em></a>. In H. Nguyen (Ed.), <em>GPU Gems 3</em> (Chapter 32). Addison-Wesley.</li>
  <li id="reference-isaac-lab">NVIDIA. (n.d.). <a href="https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html"><em>Isaac Lab documentation</em></a>.</li>
  <li id="reference-purejaxrl">Lu, C. (n.d.). <a href="https://chrislu.page/blog/meta-disco/"><em>Meta-discoveries from PureJaxRL</em></a> [Blog post].</li>
  <li id="reference-rocketsim">ZealanL. (n.d.). <a href="https://github.com/ZealanL/RocketSim"><em>RocketSim</em></a> [Computer software]. GitHub.</li>
  <li id="reference-bullet">Bullet Physics. (n.d.). <a href="https://github.com/bulletphysics/bullet3"><em>Bullet Physics SDK</em></a> [Computer software]. GitHub.</li>
  <li id="reference-gottschalk-et-al-1996">Gottschalk, S., Lin, M. C., and Manocha, D. (1996). <a href="https://dl.acm.org/doi/10.1145/237170.237244"><em>OBBTree: A hierarchical structure for rapid interference detection</em></a>. In <em>Proceedings of the 23rd Annual Conference on Computer Graphics and Interactive Techniques</em>, 171-180.</li>
  <li id="reference-sutherland-hodgman">Sutherland, I. E., and Hodgman, G. W. (1974). <a href="https://dl.acm.org/doi/10.1145/360767.360802"><em>Reentrant polygon clipping</em></a>. <em>Communications of the ACM</em>, 17(1), 32-42.</li>
  <li id="reference-tayyub-khan-2019">Tayyub, M., and Khan, G. N. (2019). <a href="https://doi.org/10.33965/ijcsis_2019140202"><em>Heterogeneous design and efficient CPU-GPU implementation of collision detection</em></a>. <em>IADIS International Journal on Computer Science and Information Systems</em>, 14(2), 25-40.</li>
  <li id="reference-dlpack">DLPack contributors. (n.d.). <a href="https://dmlc.github.io/dlpack/latest/"><em>DLPack documentation</em></a>.</li>
  <li id="reference-van-den-bergen-2004">van den Bergen, G. (2004). <em>Collision detection in interactive 3D environments</em>. Morgan Kaufmann.</li>
  <li id="reference-varga-2000">Varga, R. S. (2000). <a href="https://doi.org/10.1007/978-3-642-05156-2"><em>Matrix iterative analysis</em></a> (2nd ed.). Springer.</li>
  <li id="reference-baraff-1997">Baraff, D. (1997). <a href="https://www.cs.cmu.edu/~baraff/sigcourse/notesd1.pdf"><em>An introduction to physically based modeling: Rigid body simulation I - Unconstrained rigid body dynamics</em></a> [Course notes]. SIGGRAPH '97.</li>
</ol>
