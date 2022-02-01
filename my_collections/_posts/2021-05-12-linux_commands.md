---
layout: single
title: "Linux CLI Cheat Sheet"
twitter-image:  /assets/posts/linux-cheat-sheet/cheat-sheet_card.png
excerpt: "Some useful commands and shortcuts for linux command line."
header:
  teaser: /assets/posts/linux-cheat-sheet/cheat-sheet_card.png
  overlay_image:  /assets/posts/linux-cheat-sheet/cheat-sheet_header.png
  overlay_filter: 0.5 # rgba(255, 0, 0, 0.5), linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 255, 255, 0.5))
date:   2021-05-12 20:30:00 +0530
last_modified_at: 2021-07-02 20:30:00 +0530
categories: other linux
published: true

# table
toc: false
# toc_label: "Label goes here"
# toc_icon: "<some font awesome icon>"
# toc_sticky: true

---

Some shortcuts that I learned while working with Linux. Most of these are very basic, so if
you have even moderate experience working with the Linux shell, you probably won't find anything
new here.

# Commands

## sudo !!

Run last command with sudo privileges

## Separators

1. **;** - Separator. Run first command and then second command.
2. **&&** - Logical *and*. Run second command only if first command exits successfully.
3. **\|\|** - Logical *or*. Run second command only if first command fails.

## Create multiple folders

`mkdir -p folder/{dir1, dir2}/{dira, dirb, dirc}/{1..100}`

## See directory size

`du -sh dirname`

## Monitor system

1. See free memory - `free` with `-m` or `-g` flags
2. Process viewer - `htop`

# Keyboard shortcuts

| Shortcut      | Function                                         |
|---------------|--------------------------------------------------|
| Ctrl+C        | Terminate Process                                |
| Ctrl+D        | Exit session/disconnect from a remote connection |
| Ctrl+L        | Scroll up to clear screen                        |
| Ctrl+A        | Go to beginning of line                          |
| Ctrl+E        | Go to end of line                                |
| Alt+F/Alt+B   | Go one word forward/backward                     |
| Ctrl+R        | Reverse i search                                 |
| Ctrl+G        | Exit reverse i search                            |
| Ctrl+K/Ctrl+U | Cut to the end/beginning of the line             |
| Ctrl+Y        | Yank. Paste the cut items                        |
| Ctrl+W        | Cut one word at a time                           |

