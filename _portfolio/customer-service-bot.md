---
title: "Darry - A ChatGPT powered customer service bot"
excerpt: "Demo of a chat bot for an e-commerce website built using ChatGPT that can answer questions about products and orders in real time by interacting with the website’s database."
collection: portfolio
date: 2023-07-08 20:30:00 -0500
last_modified_at: 2023-07-08 08:30:00 -0500
---
[![Open In Colab][colab-badge]][colab-notebook] [![Github forks][gh-fork-shield]][github-repo]

Providing good customer service is one of the most important but also challenging aspects of running a successful business
that deals with a lot of customers. Ideally, a company should be able to provide 24/7 human support to its customers, but 
that can be very cost prohibitive, and difficult to scale as the business grows. This is where chatbots come in.

Companies have been using chatbots for years now to provide instant answers to common questions and free up their customer
service team to focus on more complex issues. My first experience with machine learning was building a chatbot for a
customer service use case during my undergrad internship in 2017. The chat bot I trained from using TensorFlow 1.0 and 
a large unprocessed dataset of tweets and their replies was not very good, but the technology has come a long way since then.

Since the release of ChatGPT in 2022, building such a chatbot has become a much simpler process, where the developer
doesn't have to worry about the model architecture or training process. This project demonstrates how well ChatGPT can
perform on as a customer service bot that is customized for a specific company without any fine-tuning, by carefully
tuning the prompts to the model to make it function in the desired manner.

# Basics

## OpenAI API

Most people are familiar with the web interface of ChatGPT, where you can have a conversation with the model using a 
text-based chat interface. OpenAI also provides an API that can be used to interact with the model programmatically.

You can use the API by creating an API key in your account. Using the API is not free, but OpenAI provides some free
credits to get started. 

## Setting up the system prompt

When you chat with ChatGPT on the web interface, your messages are sent to the model with an implicit role of `user`. The
previous replies by the model in the conversation are sent with the role of `assistant`. However, before the chat starts,
the model is given a prompt that sets the context for the conversation. This prompt is the same for all users and is
set by OpenAI using the special role of `system`.

When using the API, we have the option of setting the system prompt ourselves. We can provide instructions to the model
on how it should behave in the conversation before handing over the conversation to the user. This is the most important
part of setting up the chatbot, especially when we are not doing any additional fine-tuning on the model (which might not
be feasible for many small or medium-sized businesses), and requires a lot of experimentation to get right. This process
is called "prompt engineering".

## Example chat with the model

Let's look at an example conversation with the model to get a better idea of how it works. The following conversation
was generated using the default system prompt provided by OpenAI.

```
Assistant: Hello, I am a customer service bot. How can I help you today?
User: I want to return my order.
Assistant: I am sorry to hear that. What is your order number?
User: 123456
Assistant: I have initiated the return process for your order. You will receive an email with the details shortly.
User: Thank you.
Assistant: You are welcome. Is there anything else I can help you with?
User: No, that's all.
Assistant: Thank you for using our service. Have a nice day!
```


<!-- Links -->
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-notebook]: <https://github.com/kartik727/ml-projects/blob/f4663142b3268ce3ed52c6ddd92132eb957b57f9/personalized-news-summarizer/Personalized_News_Bot.ipynb> "Colab notebook"
[gh-fork-shield]: <https://img.shields.io/github/forks/kartik727/ml-projects.svg?style=social&label=Fork&maxAge=2592000>
[github-repo]: <https://github.com/kartik727/ml-projects/tree/f4663142b3268ce3ed52c6ddd92132eb957b57f9/personalized-news-summarizer> "Github repository"
[1]: <https://www.tensorflow.org/tutorials/generative/adversarial_fgsm> "Adversarial example using FGSM - TensorFlow Core"
[2]: <https://christophm.github.io/interpretable-ml-book/adversarial.html> "Adversarial Examples - Interpretable Machine Learning (christophm.github.io)"
[3]: <https://bair.berkeley.edu/blog/2017/12/30/yolo-attack/> "Physical Adversarial Examples Against Deep Neural Networks – The Berkeley Artificial Intelligence Research Blog"
