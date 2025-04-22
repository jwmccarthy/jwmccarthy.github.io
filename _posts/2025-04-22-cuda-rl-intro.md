---
title:  "Building the Ultimate Rocket League Agent [Part 1]"
mathjax: true
layout: post
categories: reinforcement learning, physics simulation, cuda
---

Rocket League is my favorite game, but with its high skill ceiling and toxic community, it also makes me feel bad. Maybe if I can train one of the best Rocket League players to ever exist to fight my battles for me, I won't have to feel bad anymore...

My plan is to develop a Reinforcement Learning agent that learns to play Rocket League through self-play in a simulated environment. [Seer][seer] and [Nexto][nexto], the most capable Rocket League bots at the moment, were also trained using this method, so credit to them for setting the example. To build on top of this approach, I plan to implement a highly-parallel Rocket League physics simulator to run on the GPU, thereby speeding up the training process potentially by orders of magnitude.


### Available Environments

The most significant bottleneck in training a Rocket League agent is the simulation speed. [RLGym][rlgym], the current standard reinforcement learning environment, offers a 10-50x speedup over the native simulation speed of 120 Hz. This has been sufficient given days, weeks, or even months of wall clock training time, but in order to reach higher levels of training time and performance with consumer hardware (i.e. RTX 3060), we need another approach.

[RocketSim][rocketsim] is a potential alternative that offers a significant speedup over RLGym by implementing only the physics component of Rocket League, sidestepping the compute required to render everything out. According to the creators, it can simulate at roughly 100,000 Hz on just a single thread, which would allow for the processing of 20 minutes of game time in a just 1 second. [Python][rocketpy1] [wrappers][rocketpy2] exist that would allow for compatibility with most reinforcement learning frameworks out there, including [my own][jarl], but what if I wanted to go even faster?

Getting to the point, I was inspired by this [blog post][purejaxrl] to implement the physics of RocketSim in a parallelized way via CUDA, such that hundred or even thousands of Rocket League simulations could be processed simultaneously. Theoretically, assuming ~500 distinct simulations could run concurrently and a more conservative single-thread processing speed of 10,000 Hz, just 1 second of wall time could enable the collection of over 10 hours of game time experience in just a single second.

### GPU Acceleration

#### Benefits

The clearest benefit to creating such a highly-parallelized environment as this is the speed. As outlined above, absolutely enormous amounts of experience could be collected in very little time. This enables the rapid training of reinforcement learning agents, but also the ability to do more subtle and interesting experiments as detailed in the [PureJaxRL][purejaxrl] post. These include more typical adjustments such as hyperparameter tuning, but also value function evolution, which applies a genetic algorithm to the training of the value function for each individual agent as they play and learn.

A further benefit, which will also contribute significantly to the reduction in training time, is the reduction in memory copies from the CPU to GPU and vice versa. Often, reinforcement learning environments run on the CPU, either in a single-threaded or multi-threaded manner, or on separate machines on a network that cooperate to pool together experiences collected individually. If the policy neural network (or other accompanying networks that assist in the learning process) is kept on the GPU, then we need to copy our collected experiences to the GPU in order to perform updates.

By running simulations on the GPU, all data generated will be accessible to our networks automatically without the need for expensive memory copies between devices. This should again offer a significant speedup over current training methods!

#### Constraints

GPU architectures are optimized for the concurrent execution of many embarassingly-parallel operations. Matrix multiplications are a great example of tasks that can be heavily decomposed into many simple operations. 

On the surface, Rocket League appears to be quite a complex system, contrary to the typical computations allocated for GPU processing. It turns out, however, that the game state of Rocket League is fairly simple.

[seer]:      https://www.youtube.com/@UltrawideGC/videos
[nexto]:     https://github.com/Rolv-Arild/Necto
[rlgym]:     https://rlgym.org/
[purejaxrl]: https://chrislu.page/blog/meta-disco/
[rocketsim]: https://github.com/ZealanL/RocketSim/
[rocketpy1]: https://github.com/mtheall/RocketSim
[rocketpy2]: https://github.com/uservar/pyrocketsim
[jarl]:      https://github.com/jwmccarthy/JARL