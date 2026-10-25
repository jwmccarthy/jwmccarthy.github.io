---
layout: post
title: "CARL: <strong>C</strong>UDA-<strong>A</strong>ccelerated <strong>R</strong>ocket <strong>L</strong>eague"
repository: https://github.com/jwmccarthy/CARL
date: 2026-10-20
mathjax: true
---

![Carl from Jimmy Neutron](/images/carl.webp){: .post-hero-image }


This post introduces CARL, a GPU-resident Rocket League simulator built for reinforcement learning. Inspired by the likes of [IsaacLab](https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html) <a href="#reference-isaac-lab">[1]</a> and [PureJaxRL](https://chrislu.page/blog/meta-disco/) <a href="#reference-purejaxrl">[2]</a>, CARL's parallel environment enables the rapid training of Rocket League agents. Many thousands of games can be run simultaneously on a single GPU, which exposes the policy to significantly more experience in less time than serial simulations. Credit goes to [RocketSim](https://github.com/ZealanL/RocketSim) <a href="#reference-rocketsim">[3]</a> and the [Bullet physics engine](https://github.com/bulletphysics/bullet3) <a href="#reference-bullet">[4]</a> for the ported logic.

## Benchmarks

PPO was run on both CARL and [RLGymPPO C++](https://github.com/ZealanL/RLGymPPO_CPP) <a href="#reference-rlgymppo-cpp">[5]</a> to evaluate training throughput. Benchmark runs were configured with the same high-level environment, rollout target, policy/critic sizes, and PPO hyperparameters, targeting roughly 1M decision steps per measurement. Given the benchmarks were run via self-play within the 1v1 game mode, each environment step produces two actor decisions. Whole vector rollouts make individual totals vary slightly around the target, but both frameworks count one action for one car as one decision step.

Given that this benchmark includes policy forward/backward, critic forward/backward, data transfer overhead, advantage estimates, and so on, we are able to paint a complete picture of the benefits of leveraging CARL on real-world training workloads.

<table class="benchmark-table">
  <thead>
    <tr>
      <th style="text-align: center">Framework</th>
      <th style="text-align: center">Actors</th>
      <th style="text-align: center">Collection steps/s</th>
      <th style="text-align: center">Total steps/s</th>
      <th style="text-align: center">Speedup vs. C++</th>
    </tr>
  </thead>
  <tbody>
    <tr class="benchmark-cpp"><td class="framework" rowspan="6"><b>RLGymPPO C++</b></td><td>256</td><td>26,636</td><td>23,214</td><td>-</td></tr>
    <tr class="benchmark-cpp"><td>512</td><td>42,916</td><td>35,396</td><td>-</td></tr>
    <tr class="benchmark-cpp"><td>1,024</td><td>79,560</td><td>62,336</td><td>-</td></tr>
    <tr class="benchmark-cpp"><td>2,048</td><td>114,812</td><td>82,374</td><td>-</td></tr>
    <tr class="benchmark-cpp"><td>4,096</td><td>138,834</td><td>95,754</td><td>-</td></tr>
    <tr class="benchmark-cpp"><td>8,192</td><td>127,869</td><td>90,791</td><td>-</td></tr>
    <tr class="benchmark-carl-3060"><td class="framework" rowspan="6"><b>CARL (RTX 3060)</b></td><td>1,024</td><td>217,160</td><td>188,194</td><td>1.97x</td></tr>
    <tr class="benchmark-carl-3060"><td>2,048</td><td>353,144</td><td>281,998</td><td>2.94x</td></tr>
    <tr class="benchmark-carl-3060"><td>4,096</td><td>659,554</td><td>464,250</td><td>4.85x</td></tr>
    <tr class="benchmark-carl-3060"><td>8,192</td><td>1,069,439</td><td>634,151</td><td>6.62x</td></tr>
    <tr class="benchmark-carl-3060"><td>16,384</td><td>1,644,364</td><td>799,853</td><td>8.35x</td></tr>
    <tr class="benchmark-carl-3060"><td>32,768</td><td>1,994,303</td><td>890,133</td><td>9.30x</td></tr>
    <tr class="benchmark-carl-5090"><td class="framework" rowspan="6"><b>CARL (RTX 5090)</b></td><td>2,048</td><td>724,375</td><td>645,629</td><td>6.74x</td></tr>
    <tr class="benchmark-carl-5090"><td>4,096</td><td>1,098,495</td><td>880,430</td><td>9.19x</td></tr>
    <tr class="benchmark-carl-5090"><td>8,192</td><td>2,537,208</td><td>1,687,349</td><td>17.62x</td></tr>
    <tr class="benchmark-carl-5090"><td>16,384</td><td>6,515,211</td><td>2,864,158</td><td>29.91x</td></tr>
    <tr class="benchmark-carl-5090"><td>32,768</td><td>11,782,742</td><td>3,423,572</td><td>35.75x</td></tr>
    <tr class="benchmark-carl-5090"><td>65,536</td><td>17,132,375</td><td>4,813,154</td><td>50.26x</td></tr>
  </tbody>
</table>

<p class="table-caption">Benchmark code: <a href="https://github.com/jwmccarthy/carl-benchmarks">carl-benchmarks</a></p>

<details markdown="1">
<summary>Benchmark Hyperparameters</summary>

<table class="benchmark-hyperparams" style="width: 50%">
  <thead>
    <tr><th>Parameter</th><th>Value</th></tr>
  </thead>
  <tbody>
    <tr><td>Algorithm</td><td>PPO</td></tr>
    <tr><td>Total timesteps</td><td>1M</td></tr>
    <tr><td>Epochs</td><td>1</td></tr>
    <tr><td>Minibatch size</td><td>25,000</td></tr>
    <tr><td>Total batch size</td><td>100,000</td></tr>
    <tr><td>Frameskip</td><td>8</td></tr>
    <tr><td>Max ticks per episode</td><td>36,000</td></tr>
  </tbody>
</table>
</details>

Collection and total throughput were measured in decision steps per second, where "collection" excludes PPO updates, capturing just the speed of the simulation and forward pass, and "total" includes the time for PPO updates to occur.

<figure class="post-figure post-figure-throughput">
  <img src="{{ '/images/unified-ppo-throughput-log-x-y.svg' | relative_url }}" alt="Unified PPO collection and total decision-step throughput by actors for RLGymPPO C++ and CARL">
  <figcaption>Unified one-epoch PPO benchmark. Solid lines measure collection throughput; dotted lines include PPO updates.</figcaption>
</figure>

Benchmarks were run locally on an Intel Core i5-13600K CPU, NVIDIA GeForce RTX 3060 GPU with 12 GiB VRAM, and 32 GiB RAM, as well as a Runpod instance with an RTX 5090.

I'll also note that CARL is still relatively unoptimized end-to-end, so any advice or suggestions are welcome! I feel these speedup multiples can be pushed even further.

## Interface
CARL exposes the underlying simulation via the standard Gymnasium API. It is planned to add support for other ML libraries like JAX and TensorFlow, but the default wrapper at the moment is written in PyTorch. As with all other Gymnasium-wrapped environments, one need only initialize, reset, and step!
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

### Custom Rewards
Rewards are computed on-device through `RewardContext`. A reward function receives the previous state and present transition, observations, actions, game events, and episode metadata. It returns one reward per car with shape $(n_{\mathrm{sim}}, n_{\mathrm{cars}})$. Multiple reward functions are summed and multiplied by `reward_scale`. Registering custom rewards replaces the default score-only reward.

As an example, we can reward based on car velocity (higher-the-better):

```python
import carl

def speed_reward(context):
    return context.current.car_velocity.norm(dim=-1) / carl.CAR_MAX_SPEED

env = CARLTorchVectorEnv(
    n_sim=1024,
    n_blue=1,
    n_orange=1,
    reward_funcs=[speed_reward],
)
```

An arbitrary list of custom rewards may be applied, though it is usually best practice to unify rewards depending on their requisite context.

## Implementation

As stated in the introduction, much of the logic in CARL is redundant with that of RocketSim/Bullet. However, a few alterations were made to make CARL a better fit for execution on the GPU. This section highlights those key differences.

### Game State
The game state resides entirely on the GPU in a struct-of-arrays layout. During simulation, the kernels index into these arrays to perform data-dependent computations or modify state information.

<figure class="post-figure">
  <img src="{{ '/images/aos-soa-memory-layout.png' | relative_url }}" alt="Comparison of array-of-structures and struct-of-arrays memory layouts">
  <figcaption>Array-of-structures and struct-of-arrays memory layouts. <a href="#reference-zhao-et-al-2022">[6]</a></figcaption>
</figure>

SoA is the optimal approach for data storage in GPU applications due to the enforcement of coalesced fetch instructions for contiguous elements. If we were instead to store arrays of structs, each associated with a single game's state, the GPU would be forced to perform strided accesses, leading to more memory transactions. We'd also be pulling in full structs worth of data we may not need at the time.

### Arena Geometry
The arena mesh is loaded into global memory upon environment initialization. In contrast to RocketSim's [bounding volume hierarchy (BVH)](https://en.wikipedia.org/wiki/Bounding_volume_hierarchy) broad phase <a href="#reference-rocketsim">[3]</a>, CARL divides the arena into a 3D broad phase grid of fixed size. The broad phase helps to reduce the number of triangle tests during collision resolution, and the fixed grid in particular enforces greater regularity of the triangle checks on the GPU.

CARL leverages the prefix sum construction described in NVIDIA's [Broad-Phase Collision Detection with CUDA](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda) <a href="#reference-le-grand-2007">[7]</a>. The prefix sum converts each broad phase cell's triangle occupancy into offsets/counts, allowing the car-arena and ball-arena collision kernels to determine where a cell's triangle list begins and how many triangles it must test.

<figure class="post-figure post-figure-compact post-figure-grid">
  <img src="{{ '/images/gpu-gems-cell-id-array.jpg' | relative_url }}" alt="Broad-phase cell ID array after construction">
  <figcaption>Initial contents of the broad-phase cell ID array. <a href="#reference-le-grand-2007">[7]</a></figcaption>
</figure>

It is quick to check both sphere and axis-aligned bounding box (AABB) occupancy, so we can quickly cull most of the triangles in the arena before checking specific tris against the car and ball.

### SAT Narrow Phase
Unlike RocketSim, CARL does not use [GJK](https://en.wikipedia.org/wiki/Gilbert%E2%80%93Johnson%E2%80%93Keerthi_distance_algorithm)/[EPA](https://en.wikipedia.org/wiki/Expanding_polytope_algorithm) <a href="#reference-van-den-bergen-2004">[8]</a>. Instead, it implements the [Separating Axis Theorem (SAT)](https://en.wikipedia.org/wiki/Hyperplane_separation_theorem#Separating_axis_theorem) <a href="#reference-van-den-bergen-2004">[8]</a> to identify car-arena and car-car collisions without iterative support mapping.

<figure class="post-figure post-figure-half post-figure-sat">
  <img src="{{ '/images/separating-axis-theorem.webp' | relative_url }}" alt="Separating Axis Theorem overlap test">
  <figcaption>Separating Axis Theorem OBB overlap test. <a href="#reference-tayyub-khan-2019">[9]</a></figcaption>
</figure>

Sphere-OBB and sphere-triangle closest point tests generate car-ball and ball-arena contacts. Clipping via [Sutherland-Hodgman](https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm) <a href="#reference-sutherland-hodgman">[10]</a> constructs the car contact manifolds.

<figure class="post-figure post-figure-half">
  <img src="{{ '/images/sutherland-hodgman-clipping.svg' | relative_url }}" alt="Steps of Sutherland-Hodgman polygon clipping">
  <figcaption>Sutherland-Hodgman polygon clipping. <a href="#reference-sutherland-hodgman">[10]</a> Illustration by <a href="https://commons.wikimedia.org/wiki/File:Sutherland-Hodgman_clipping_sample.svg">Wojciech mula</a>, public domain.</figcaption>
</figure>

These manifolds are persistent and updated each tick depending on the generation of new contact points. A maximum of 4 points is allowed per manifold in line with RocketSim.

### I/O
Below the Gymnasium wrapper, `carl.Env` is a CUDA-only [DLPack](https://dmlc.github.io/dlpack/latest/) interface. <a href="#reference-dlpack">[11]</a> Inputs must be contiguous tensors on the device. This way, no host round trip is required.

The native batch is organized by match rather than actor. All cars in a single simulation share one physics world, score, reward event, and termination/truncation flags. The PyTorch wrapper flattens the leading two dimensions so that each car appears as a separate environment.

`step()` first packs the transition state and terminal observations, then automatically resets completed matches to shuffled kickoffs (or samples from user-specified reset states) and produces the next observation buffers. Returned DLPack tensors are non-owning views over CARL's reusable GPU buffers, so callers that need to retain a result across a subsequent `step()` must clone or copy it. The Gymnasium PyTorch wrapper does this by default and expands match-level rewards and terminations back to each participating car.

## Conclusion
CARL represents a significant leap in Rocket League RL throughput, and it is my opinion that this is largely a good and valuable thing for the community. Thank you for reading if you made it this far! All love to Mr. Carl Wheezer.

<b>Note:</b> If anyone who is familiar with the RLGym ecosystem finds any fault with the benchmarking methodology, please reach out! I'm confident CARL enables an appreciable training speed-up, but I'd like to ensure all comparisons are appropriate.

## References

 <ol>
  <li id="reference-isaac-lab">NVIDIA. (n.d.). <a href="https://isaac-sim.github.io/IsaacLab/v3.0.0-EA/index.html"><em>Isaac Lab documentation</em></a>.</li>
  <li id="reference-purejaxrl">Lu, C. (n.d.). <a href="https://chrislu.page/blog/meta-disco/"><em>Meta-discoveries from PureJaxRL</em></a> [Blog post].</li>
  <li id="reference-rocketsim">ZealanL. (n.d.). <a href="https://github.com/ZealanL/RocketSim"><em>RocketSim</em></a> [Computer software]. GitHub.</li>
  <li id="reference-bullet">Bullet Physics. (n.d.). <a href="https://github.com/bulletphysics/bullet3"><em>Bullet Physics SDK</em></a> [Computer software]. GitHub.</li>
  <li id="reference-rlgymppo-cpp">ZealanL. (n.d.). <a href="https://github.com/ZealanL/RLGymPPO_CPP"><em>RLGymPPO C++</em></a> [Computer software]. GitHub.</li>
  <li id="reference-zhao-et-al-2022">Zhao, W.-L., Wang, W., and Wang, Q. (2022). <a href="https://doi.org/10.1007/s11227-021-04153-0"><em>Optimization of cosmological N-body simulation with FMM-PM on SIMT accelerators</em></a>. <em>The Journal of Supercomputing</em>, 78(5), 7186-7205.</li>
  <li id="reference-le-grand-2007">Le Grand, S. (2007). <a href="https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-32-broad-phase-collision-detection-cuda"><em>Broad-phase collision detection with CUDA</em></a>. In H. Nguyen (Ed.), <em>GPU Gems 3</em> (Chapter 32). Addison-Wesley.</li>
  <li id="reference-van-den-bergen-2004">van den Bergen, G. (2004). <em>Collision detection in interactive 3D environments</em>. Morgan Kaufmann.</li>
  <li id="reference-tayyub-khan-2019">Tayyub, M., and Khan, G. N. (2019). <a href="https://doi.org/10.33965/ijcsis_2019140202"><em>Heterogeneous design and efficient CPU-GPU implementation of collision detection</em></a>. <em>IADIS International Journal on Computer Science and Information Systems</em>, 14(2), 25-40.</li>
  <li id="reference-sutherland-hodgman">Sutherland, I. E., and Hodgman, G. W. (1974). <a href="https://dl.acm.org/doi/10.1145/360767.360802"><em>Reentrant polygon clipping</em></a>. <em>Communications of the ACM</em>, 17(1), 32-42.</li>
  <li id="reference-dlpack">DLPack contributors. (n.d.). <a href="https://dmlc.github.io/dlpack/latest/"><em>DLPack documentation</em></a>.</li>
</ol>
