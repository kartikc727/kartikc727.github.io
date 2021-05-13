---
layout: single
title: "My First Jekyll Post"
header:
  teaser: /assets/images/teasers/first_post.jpg
date:   2021-05-12 20:30:00 +0530
categories: demo
published: true
---

This is a paragraph in my first post.
Show off your Markdown!

## Heading Two 

Any text with no empty lines between will become a paragraph.
Leave an blank line between headings and paragraphs.
Font can be *Italic* or **Bold**.
Code can be highlighted with `backticks`.

Hyperlinks look like this [GitHub Help](https://help.github.com/).

A bullet list is created using `*`, `+`, or `-`, like:

- dog
- cat
- muffin
- cake

A numbered list is created using a number + `.`, like:

1. one
2. two
6. three
2. four


<form method="POST" action="https://staticman-comment-kartik727.herokuapp.com/v3/entry/github/kartik727/kartik727.github.io/main/comments">
  <input name="options[redirect]" type="hidden" value="https://kartik727.github.io/">
  <!-- e.g. "2016-01-02-this-is-a-post" -->
  <input name="options[slug]" type="hidden" value="{{ page.slug }}">
  <label>Name<input name="fields[name]" type="text"></label>
  <label>E-mail<input name="fields[email]" type="email"></label>
  <label>Message<textarea name="fields[message]"></textarea></label>
  
  <button type="submit">Go!</button>
</form>