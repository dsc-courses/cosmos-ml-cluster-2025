---
layout: page
title: 👩‍🏫 Staff
description: A listing of all the course staff members.
nav_order: 4
---

# 👩‍🏫 Staff

## Instructor

{% assign instructors = site.staffers | where: 'role', 'Instructor' %}
{% for staffer in instructors %}
{{ staffer }}
{% endfor %}

## Staff

{% assign tfs = site.staffers | where: 'role', 'Teacher Fellow' %}
{% for staffer in tfs %}
{{ staffer }}
{% endfor %}

{% assign tas = site.staffers | where: 'role', 'TA' %}
{% for staffer in tas %}
{{ staffer }}
{% endfor %}

{% assign staff = site.staffers | where: 'role', 'Tutor' %}

<div class="role">
  {% for staffer in staff %}
  {{ staffer }}
  {% endfor %}
</div>
