---
layout: post
title: "RL From the Ground Up"
categories: RL, Gymnasium, PyTorch
custom_excerpt: "Intro to the blog series that will cover the development of my reinforcement learning framework, Gumbo"
---

### What Am I Even Doing?

Hi! This is a blog that I'll use to document my progress in my personal projects. As of now, my time is dedicated solely to learning and implementing reinforcement learning algorithms. To do so, I'm developing an RL framework which I've called "Gumbo" for no particular reason. I guess the modular components resemble the ingredients of a soup? Anyway, this post is just meant as an introduction to the framework, an explaination of certain dependencies, and a demonstration of the current capabilities of this framework in training autonomous agents.

### Dependencies

There are two key dependencies upon which Gumbo is built: the Gymnasium environment API and PyTorch. These are common choices in many other RL libraries, but I'll quickly go over my rationale for their usage.

The Gymnasium API was chosen due to the following reasons:
1. The interface is simple
2. Many environments have already been implemented with the same API
3. It is easy to adapt to custom environments
4. Several other environments use the same API

Likewise, PyTorch was chosen as the basis for Gumbo's data and model components for fairly straightforward reasons:
1. I have prior experience with it
2. It is easy and intuitive to use
3. I have an NVIDIA GPU

Everything else within Gumbo is built from scratch on top of these libraries. I plan to document the way I'm handling data via PyTorch tensors in a future post, as I believe I'm storing and operating on episodic data in an interesting way.

### Performance

Just as a quick demonstration of performance, here are agents trained via Gumbo and their episodic return over the course of training via PPO.

##### Lunar Lander

The lunar lander environment has the agent control four thrusters on a small craft to land it safely from random initial conditions. It is penalized for thruster usage and crashes, and rewarded significantly for successfully landing.

<p float="left" align="middle">
    <img src="/assets/lunar_lander_demo.gif">
</p>

Look at it go! The agent learns to let gravity do the work for the early part of the descent, avoiding penalties for needless thruster usage. It then engages its vertical thruster (along with some orientation corrections) at just the right time to land safely within the target zone.

Below is the moving average of the episodic return during training for five distinct training runs. The progress of the agents can vary widely, but after around 1M collected timesteps, they tend to converge on reward-maximizing behavior.

![](/assets/lunar_lander_5_runs.png)

##### Bipedal Walker

The bipedal walker environment requires that the agent output continuous values representing the motor speed for each of its four joints. The continuous control task requires a slightly different policy architecture than that of the discrete action lunar lander. I've put it here to demonstrate that Gumbo is capable of doing both in its present state. The reward function in this case is simple: the greater the distance, the more reward collected. 

<p float="left" align="middle">
    <img src="/assets/bipedal_walker_demo.gif">
</p>

This isn't quite a walk as much as it is a shuffle, but the agent has quite a pace to it! Using its back leg to stabilize and push while pulling with the front leg is a pretty clever and, as it turns out, reliable strategy for walking over variable height terrain.

Once again, here are the episodic returns for five training runs.

![](/assets/bipedal_walker_5_runs.png)

### What Now?

I have quite a long wish list of features I'd like to implement. Most of them allow me to expand the number of Gymnasium environments Gumbo can handle, such as Atari or MuJoCo. This entails things such as frame stacking, observation & action transformations, CNNs, RNNs, etc. I'd also like to implement more comprehensive logging procedures to allow for quicker debugging and performance reports. This is just the start, but I'm learning a lot creating this framework and I look forward to making it a more capable, well-oiled system!