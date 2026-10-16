---
title: "A small place to think in public"
date: 2026-10-16
excerpt: "Why I am making room for short notes, unfinished ideas, and the things I learn while building."
mathjax: true
layout: single
author_profile: false
---

Writing things down helps me notice what I am learning. This blog is a simple place to keep those notes and share the useful ones.

## GAIfO in brief

Generative Adversarial Imitation from Observation (GAIfO) is an imitation-learning algorithm for situations where we can observe an expert's states but do not have access to the expert's actions.

Instead of comparing state-action pairs, GAIfO trains a discriminator to distinguish expert state transitions, $(s_t, s_{t+1})$, from transitions produced by the learner. The learner then uses the discriminator's feedback as a reward and improves its policy.

The discriminator objective can be written as:

$$
\max_D \; \mathbb{E}_{\tau_E}[\log D(s_t, s_{t+1})] + \mathbb{E}_{\tau_\pi}[\log(1 - D(s_t, s_{t+1}))]
$$

Here, $D$ estimates whether a transition came from the expert. As the learner's transitions become more like the expert's, the discriminator can no longer reliably tell them apart.
