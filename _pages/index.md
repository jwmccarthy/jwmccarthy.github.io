---
title: "Blog"
layout: archive
permalink: /
author_profile: true
---

Notes on machine learning, building things, and what I'm learning along the way.

{% for post in site.posts reversed %}
{% include archive-single.html %}
{% endfor %}
