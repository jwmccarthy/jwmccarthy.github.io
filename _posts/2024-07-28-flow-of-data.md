---
layout: post
title: "Data Manipulation in Gumbo Illustrated"
categories: RL, PyTorch, Data, Visualization
custom_excerpt: "Visualizing how data collection, storage, and manipulation is handled in Gumbo"
---

Getting to the point: I think the way I'm handling data in Gumbo is interesting and I'd like to illustrate the details behind how it works. Below are some diagrams that exemplify the flow of data throughout the framework, demonstrating its ability to neatly handle episodic data, as well as visualizing its modular structure. Let's go!

### Data Collection

With any reinforcement learning algorithm, we will have to collect data describing policy-environment interactions. These are often split into episodes, which themselves are self-contained sequences of action-observation pairs. More specifically, the standard in RL is model data collection as a MDP of state-action-state tuples \\((s_t, a_t, s_{t+1})\\). 

These transitions are collected in a loop and each step stored in an ``EpisodicBuffer`` instance. This buffer holds data in PyTorch tensors via a ``TensorDataset``, which I have constructed as a dataclass with added tensor-like functionality. The buffer also contains ``Subsets``, representing episodes, that reference particular locations in the common tensor storage. These episodes also contain auxiliary data, such as episode statistics or, importantly, terminal observations (useful for bootstraping reward).

![Gumbo data architecture](/assets/gumbo_data_collect.png)