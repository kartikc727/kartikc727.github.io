---
layout: single
title: "Linux CLI cheat sheet"
twitter-image:  /assets/posts/linix-cheat-sheet/cheat-sheet_card.png
excerpt: "Some useful commands and shortcuts for linux command line."
header:
  teaser: /assets/posts/linix-cheat-sheet/cheat-sheet_card.png
  overlay_image:  /assets/posts/linix-cheat-sheet/cheat-sheet_header.png
  overlay_filter: 0.5 # rgba(255, 0, 0, 0.5), linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 255, 255, 0.5))
date:   2021-05-12 20:30:00 +0530
last_modified_at: 2021-07-02 20:30:00 +0530
categories: other linux
published: true

# table
toc: true
# toc_label: "Label goes here"
# toc_icon: "<some font awesome icon>"
toc_sticky: true

sidebar:
  - nav: other-posts
---

Some shortcuts that I learned while working with Linux, most of these are very basic, so if
you have a lot of experience working with the Linux shell, you probably won't find anything
new here. I have categorised them from basic to advanced based on my own best judgement.


# Basic



1. **Ctrl + C**  
Terminate a process. If you have a script running that is taking longer than expected or for some other reason you want to kill it.



2. **Ctrl + D**  
Exit session / disconnect from a remote connection that you SSH'd into.



3. **Tab**  
Auto Complete commands or filenames. Shows a list in case of multiple matches.


4. **Up arrow / Down arrow**  
Last run command. Up arrow populates your command prompt with the last run command, you can keep pressing Up to keep going back in the command history. Conversely, pressing Down arrow will do the opposite and cycle to the more recent commands.



5. **clear / Ctrl + L**  
Clear screen. Ctrl + L will scroll you up so that your current command is on the first line of the command window. Using the command `clear` will clear the window so you won’t be able to scroll up to see previous commands. Of course you can still press the Up arrow to cycle through your command history.


# Intermediate

1. **Ctrl + A**  
Go to the beginning of the line. If you forget to write `sudo` before a command that requires elevated privileges, instead of using the left arrow to go back to the start of the line, you can press Ctrl + A to get to the beginning in one shot.



2. **Ctrl + E**  
End of the line. Opposite of Ctrl + A. After writing `sudo` at the start of the command, you can press Ctrl + E to go back to the end and finish typing your command.



3. **Ctrl + C**  
Get out of writing a command. If you want to cancel typing some command, you can press Ctrl + C to get to a new fresh line and start doing something else.



4. **Alt + F / Alt + B**  
Navigating the command word-by-word. Alt + F to go forward one word, Alt + B to go back one word



5. **Ctrl + R**  
Reverse i search.  Press Ctrl + R to go in reverse i search mode, type a substring of the command you want to search, you’ll get the most recent command that matches your search string. You can hit Ctrl + R again to go to the next most recent search and so on.



6. **Ctrl + G**  
Go back to whatever you were typing (preserves text). You can press Ctrl + G to exit reverse i search and your previously written command will be preserved.


# Advanced

1. **sudo !!**  
Run the last run command with sudo.



2. **Ctrl + K**  
Cut to the end of the line (Kill). If your cursor is in the middle of a command, Ctrl + K will delete whatever is written after it.



3. **Ctrl + U**  
Cut to the beginning. Complementary to Ctrl + K.



4. **Ctrl + Y**  
Yank. Paste whatever you cut using Ctrl + K / Ctrl + U



5. **Ctrl + W / Alt  + Backspace**  
Cut one word at a time. Words cut using Ctrl + W can be yanked back, those cut with Alt + Backspace can’t.


# References

[tutoriaLinux (YouTube)](https://www.youtube.com/c/tutoriaLinux/featured)

[Joe Collins (YouTube)](https://www.youtube.com/user/BadEditPro)
