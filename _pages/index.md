---
title: "Blog"
layout: home
permalink: /
author_profile: true
---

Notes on machine learning, building things, and what I'm learning along the way.

{% assign current_year = "" %}
{% for post in site.posts %}
  {% assign post_year = post.date | date: "%Y" %}
  {% if post_year != current_year %}
    {% unless forloop.first %}</ul>{% endunless %}
    <h2 class="archive-year">{{ post_year }}</h2>
    <ul class="blog-list">
    {% assign current_year = post_year %}
  {% endif %}
  <li><a href="{{ post.url | relative_url }}">{{ post.title }}</a> <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%d %b" }}</time></li>
{% endfor %}
</ul>
