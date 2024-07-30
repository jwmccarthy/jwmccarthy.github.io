---
layout: post
title: "Making Gumbo (Not Soup)"
categories: RL, PyTorch, Optimization
custom_excerpt: "Intro to the blog series that will cover the development of my reinforcement learning framework, Gumbo"
---

### What Am I Even Doing?

Hi! This is a blog that I'll use to document my progress in my personal projects. As of now, my time is dedicated solely to learning and implementing reinforcement learning algorithms. To do so, I'm developing an RL framework which I've called "Gumbo" for no particular reason. I guess the modular components resemble the ingredients of a soup? Anyway, this post is just meant as an introduction to the framework, an explaination of certain dependencies, and a demonstration of the current capabilities of this framework in training autonomous agents.

### Dependencies

There are two key dependencies upon which Gumbo is built: the Gymnasium environment API and PyTorch. These are common choices in many other RL libraries, but I'll quickly go over my rationale for their usage.

The Gymnasium API was chosen due to the following reasons:
1. The interface is simple
2. Many environments have already been implemented in it
3. It is easy to adapt to custom environments
4. Several other environments use the same API

Likewise, PyTorch was chosen as the basis for Gumbo's data and model components for fairly straightforward reasons:
1. I have prior experience with it
2. It is easy and intuitive to use
3. I have an NVIDIA GPU

Everything else within Gumbo is built from scratch on top of these libraries. I plan to document the way I'm handling data via PyTorch tensors in a future post, as I believe I'm storing and operating on episodic data in an interesting way.

### Performance

Just as a quick demonstration of performance, here are agents trained via Gumbo and their episodic return over the course of training.

#### Lunar Lander

<p align="middle" float="left">
    <img src="/assets/lunar_lander_demo.gif" width="400"/>
    <img src="/assets/lunar_lander_5_runs.png" width="400"/>
</p>