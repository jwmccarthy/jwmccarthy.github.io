---
layout: post
title: "Data Manipulation in Gumbo Illustrated"
categories: RL, PyTorch, Data, Visualization
custom_excerpt: "Visualizing how data collection, storage, and manipulation is handled in Gumbo"
---

Getting to the point: I think the way I'm handling data in Gumbo is interesting and I'd like to illustrate the details behind how it works. Below are some diagrams that exemplify the flow of data throughout the framework, demonstrating its ability to neatly handle episodic data, as well as visualizing its modular structure. Let's go!

### Data Collection

With any reinforcement learning algorithm, we will have to collect data describing policy-environment interactions. These are often split into episodes, which themselves are self-contained sequences of action-observation pairs. More specifically, the standard in RL is to model data collection as an MDP of state-action-state tuples \\((s_t, a_t, s_{t+1})\\), each of which has an associated reward obtained from the environment. 

These transitions are collected in a loop and at each step stored in an ``EpisodicBuffer`` instance. This buffer holds data in PyTorch tensors via a ``TensorDataset``, which I have constructed as a dataclass with added tensor-like functionality. Additionally, this buffer contains episode ``Subsets``, created upon the termination of an episode, that reference particular locations in the common tensor storage. These episodes also contain auxiliary data, such as episode statistics or, importantly, terminal observations (useful for bootstraping reward).

In off-policy methods we will keep experiences from previous training iterations, so in those cases the buffer will have memory yet to be assigned as indicated by red region in the diagram. On-policy methods will fill the entire buffer prior to each training iteration.

![EpisodicBuffer usage](/assets/gumbo_data_1.png)

The episodic nature of the buffer comes in handy for both logging model performance (episode lengths, cumulative reward, etc.) and certain derived attribute calculations. In particular, advantage estimation requires the terminal observation in the event of an episode truncation to accurately estimate future returns.

### Data Augmentation

This section will demonstrate the training data augmentation process for PPO. Any data specific to the RL method implemented via Gumbo will have to specify the procedures that generate the data required for said method.

First we can obtain quantities that are calculable across the entire dataset: action log probabilities & values. Observations are fed into the policy module to obtain the logits that define the action distribution. This distribution is then used to compute the log probabilities of each action in the dataset. Values are obtained as the direct output from some value estimation module which again takes the observation tensor as input.

![Augmenting full tensors](/assets/gumbo_data_2.png)

Calling ``get_data()`` on our buffer returns the data within said buffer which is sliced to contain only the data up to the latest encountered index. This function returns a copy of the underlying data, so any additional attributes will not appear in the buffer after augmenting our dataset. However, PyTorch tensor slices will refer to the same underlying memory, so we aren't performing redundant memory copies (which is especially important when executing on the GPU).

Next, we leverage the episodic nature of our data buffer to calculate the advantages.

![Episodic training data augmentation](/assets/gumbo_data_2.png)
