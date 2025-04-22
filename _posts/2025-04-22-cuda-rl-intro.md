---
title:  "Building the Ultimate Rocket League Agent [Part 1]"
mathjax: true
layout: post
categories: reinforcement learning, physics simulation, cuda
---

Rocket League is my favorite game, but with its high skill ceiling and toxic community, it also makes me feel bad. Maybe if I can train one of the best Rocket League players to ever exist to fight my battles for me, I won't have to feel bad anymore...


My plan is to develop a Reinforcement Learning agent that learns to play Rocket League through self-play in a simulated environment. [Seer][seer] and [Nexto][nexto], the most capable Rocket League bots at the moment, were also trained using this method, so credit to them for setting the example. To build on top of this approach, I plan to implement a highly-parallel Rocket League physics simulator to run on the GPU, thereby speeding up the training process potentially by orders of magnitude.

## Available Environments

The most significant bottleneck in training a Rocket League agent is the simulation speed. [RLGym][rlgym], the current standard reinforcement learning environment, offers a 10-50x speedup over the native simulation speed of 120 Hz. This has been sufficient given days, weeks, or even months of wall clock training time, but in order to reach higher levels of training time and performance with consumer hardware (i.e. RTX 3060), we need another approach.

[RocketSim][rocketsim] is a potential alternative that offers a significant speedup over RLGym by implementing only the physics component of Rocket League, sidestepping the compute required to render everything out. According to the creators, it can simulate at roughly 100,000 Hz on just a single thread, which would allow for the processing of 20 minutes of game time in a just 1 second. [Python][rocketpy1] [wrappers][rocketpy2] exist that would allow for compatibility with most reinforcement learning frameworks out there, including [my own][jarl], but what if I wanted to go even faster?

Getting to the point, I was inspired by this [blog post][purejaxrl] to implement the physics of RocketSim in a parallelized way via CUDA, such that hundred or even thousands of Rocket League simulations could be processed simultaneously. Theoretically, assuming ~500 distinct simulations could run concurrently and a more conservative single-thread processing speed of 10,000 Hz, just 1 second of wall time could enable the collection of over 10 hours of game time experience in just a single second.

## GPU Acceleration

### Benefits

The clearest benefit to creating such a highly-parallelized environment as this is the speed. As outlined above, absolutely enormous amounts of experience could be collected in very little time. This enables the rapid training of reinforcement learning agents, but also the ability to do more subtle and interesting experiments as detailed in the [PureJaxRL][purejaxrl] post. These include more typical adjustments such as hyperparameter tuning, but also value function evolution, which applies a genetic algorithm to the training of the value function for each individual agent as they play and learn.

A further benefit, which will also contribute significantly to the reduction in training time, is the reduction in memory copies from the CPU to GPU and vice versa. Often, reinforcement learning environments run on the CPU, either in a single-threaded or multi-threaded manner, or on separate machines on a network that cooperate to pool together experiences collected individually. If the policy neural network (or other accompanying networks that assist in the learning process) is kept on the GPU, then we need to copy our collected experiences to the GPU in order to perform updates.

By running simulations on the GPU, all data generated will be accessible to our networks automatically without the need for expensive memory copies between devices. This should again offer a significant speedup over current training methods!

### Constraints

GPU architectures are optimized for the concurrent execution of many embarassingly-parallel operations. Matrix multiplications are a great example of tasks that can be heavily decomposed into many simple operations. 

#### Physics State

On the surface, Rocket League appears to be quite a complex system, contrary to the typical computations allocated for GPU processing. It turns out, however, that the game state of Rocket League is fairly small. 

Here's a minimal look at the car states that require updates each tick:

| Field | Data Type | Size (bytes) | Description |
|-------|-----------|--------------|-------------|
| Position | Vector | 16 | Position in world space |
| Rotation Matrix | Matrix | 48 | Car orientation |
| Linear Velocity | Vector | 16 | Speed and direction of linear motion |
| Angular Velocity | Vec | 16 | Speed and direction of rotation |
| Suspension Length | float[4] | 16 | Expansion/Contraction distance of suspension |
| Jump Flag | bool | 1 | Whether the car has jumped |
| Time Since Jump | float | 4 | Time since jump input |
| Flip Flag | bool | 1 | Whether the car is currently flipping |
| Supersonic Flag | bool | 1 | Whether the car is currently supersonic |
| **Total** | | **119** | Accounts for struct padding |

For up to 8 cars, this still comes in under 1 KB. Assuming similar memory requirements for other state information such as boost pad status, ball physics state, etc., we still only require a handful of KBs, which is entirely manageable for CUDA threads.

#### Arena Mesh

I conveniently left out the memory requirements for the arena mesh in the section above, but this is because there is a nice way to handle meshes that are constant between threads: CUDA texture memory.

Given that the arena geometry is the same among all the games of Rocket League that will be happening in parallel, we can store a single instance of the mesh in CUDA's texture memory to reduce memory consumption and allow for concurrent access between threads.

#### Kernel Size

I don't think I should run into CUDA's kernel size constraints, but it would be nice to split this simulation environment into multiple kernels that run in sequence, both stylistically and for debugging purposes. At the moment, given what I understand about RocketSim and Bullet physics, this might be a good way to deliniate the physics kernels:

 1. __Dynamic collision broad phase__ - Identify pairs of dynamic objects (cars, ball) that might be colliding.

 2. __Static collision broad phase__ - Identify pairs of dynamic objects and subsets of static arena mesh that may be colliding.
 
 3. __Suspension collision broad phase__ - Identify possibility of wheel contacts and suspension updates.

 4. __Dynamic collision narrow phase__ - Accumulate contact points between dynamic bodies.

 5. __Static collision narrow phase__ - Accumulate contact points between dynamic objects and static geometry.

 6. __Suspension updates__ - Accumulate wheel contacts and suspension compression.

 7. __Sequentially solve impulses__ - Accumulate forces from contact points and sequentially solve impulses to apply to dynamic rigid bodies.

It might make more sense to split these out into device functions to run in a single kernel from an optimization perspective, so I may reevaluate when I get to a point where I can evaluate the pros and cons of such a decision.

## Current Progress

### Onboarding

I've been using a few tools to navigate the RocketSim/Bullet physics codebase, which has helped me grasp the steps strictly necessary for simulating Rocket League physics. I'm starting from zero when it comes to complex simulations like this, so these have been invaluable in getting up to speed:

#### [Sourcetrail][srctrail]
A source explorer that provides a graph-based UI for navigating dependencies within large codebases.

#### [Claude Code][claude]
I can't help but think that using an LLM for code exploration is cheating a bit, but it's helpful to ask it questions for open-ended navigation of the codebase. I'm diligent about not asking it to code for me though, because the point of this project (alongside creating a Rocket League demon that will annihilate my enemies) is to learn!

### Build System

I will be using CMake along with PyBind11 to compile my C++/CUDA code into a Python module. The API of the environment I develop will resemble the step-reset structure of [OpenAI Gym][openaigym]/[Gymnasium][gymnasium] environments, so it will easily integrate directly with existing RL frameworks.

## Next Steps

The only thing left to do is to build the thing. I'll update this blog as I reach certain milestones in the development process.


[seer]:      https://www.youtube.com/@UltrawideGC/videos
[nexto]:     https://github.com/Rolv-Arild/Necto
[rlgym]:     https://rlgym.org/
[purejaxrl]: https://chrislu.page/blog/meta-disco/
[rocketsim]: https://github.com/ZealanL/RocketSim/
[rocketpy1]: https://github.com/mtheall/RocketSim
[rocketpy2]: https://github.com/uservar/pyrocketsim
[jarl]:      https://github.com/jwmccarthy/JARL
[srctrail]:  https://github.com/CoatiSoftware/Sourcetrail
[claude]:    https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview
[openaigym]: https://www.gymlibrary.dev/index.html
[gymnasium]: https://gymnasium.farama.org/