---
layout: post
title: "Cooking Up RL (pt. 1) - Intro to Gumbo"
categories: RL, PyTorch, Optimization
custom_excerpt: "An introduction to my blog series that will cover the development of my reinforcement learning framework called Gumbo"
---

### What Am I Even Doing?

Hello! Thanks for stopping by. Apologies in advance for the tonal and thematic scratchwork that is my writing. I don't expect it'll ever get much better, but who's to say what better is anyway?

This is my first post but it finds us in the middle of the action. I'm currently working on building a reinforcement learning framework from scratch using PyTorch as the basis for all the components. RL caught my eye with the recent influx of machine learning bots in Rocket League. It seemed like a fun project to get up to speed with the latest methods in reinforcement learning (I am an optimist). Rather than implement various algorithms in existing frameworks, I wanted to create my own. This post will outline a few features of the framework as it exists today.

Proximal Policy Optimization (PPO) is among the current SOTA RL algorithms, and it was used to train the [first generation of ML Rocket League agents](https://github.com/Rolv-Arild/Necto). Many of the design decisions cater to the PPO learning procedure as it's the first I've implemented. If you see any glaring issues with the current structure as it pertains to other algorithms, I know they're in there somewhere, and one day the tech debt gods will smite me.

---

### Experience Data

None of the frameworks I investigated prior to building my own handled the collection and processing of training data in a way that I liked. A common solution was lightweight dataclasses, which are simple but lack functionality. Others were closer to what I wanted. [TorchRL](https://pytorch.org/rl/stable/index.html), for instance, uses what they call a TensorDict, which is a Python dictionary that allows for certain tensor operations.

My solution was to create a custom class that acted as a hybrid between these two methods. This [``TensorDataset``](https://github.com/jwmccarthy/Gumbo/blob/main/gumbo/data/datasets.py) allows for dot attribute access & setting, tensor operations & indexing, dataset unions, and device allocation. The accompanying ``Subset`` serves as a way to partition the broader data into slices that can represent individual episodes while still referencing a common underlying memory. It also stores auxiliary information, such as the final observation following episode truncation, which is important for bootstrapping.

The [``EpisodicBuffer``](https://github.com/jwmccarthy/Gumbo/blob/main/gumbo/data/buffers.py) class serves as the wrapper for managing ``TensorDataset`` instances as it pertains to experience collection. When an episode terminates, a ``Subset`` instance is created that can be used to reference data associated with said episode. Operations (such as advantage estimation) can then be performed in an episode-wise fashion.

---

