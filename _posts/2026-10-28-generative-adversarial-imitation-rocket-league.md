---
layout: post
title: "<b>[WIP]</b> Generative Adversarial Learning from Observation for Rocket League"
repository: https://github.com/jwmccarthy/CARL
date: 2026-10-29
mathjax: true
---

![Robot hand imitating human reference - stylized](/images/imitation-learning-robot.png){: .post-hero-image }

So many of the best Rocket League bots are extremely competent at a wide range of behaviors, but their play style is distinctly robotic. It's possible through enough tuning of the curriculum and reward specification to hand-craft more human like actions, but this can be tedious. Through imitation learning, one can train a policy to replicate expert demonstrations, thereby obtaining a more "human" behavioral aesthetic. In particular, this post outlines the (as far as I know) novel application of generative imitation learning methods to Rocket League.

## Imitation Learning

