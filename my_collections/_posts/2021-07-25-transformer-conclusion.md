---
layout: single
title: "Transformer Conclusion"
twitter-image: /assets/posts/transformer/cards/transformer-conclusion_card.png
excerpt: "A brief summary of the transformer model and how its components work together in the context of neural machine translation."
header:
  teaser: /assets/posts/transformer/cards/transformer-conclusion_card.png
  overlay_image: /assets/posts/transformer/cards/transformer-conclusion_header.png
  overlay_filter: 0.5 # rgba(255, 0, 0, 0.5), linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 255, 255, 0.5))

date:   2021-07-25 20:50:00 +0530
last_modified_at: 2021-07-25 20:50:00 +0530
categories: ML Transformer
published: true

# table
toc: true
# toc_label: "Label goes here"
# toc_icon: "<some font awesome icon>"
toc_sticky: true

# sidebar
sidebar:
  - title: "Transformer Models"
  - image: /assets/images/ml/transformer-sidebar.png
    image_alt: "Transformer Image"

  - title: All posts
  - nav: ml-posts

  - title: See also
  - nav: other-posts
---



# Overview

Let’s look at the entire process in one place.


1. The input sentence is turned into embeddings.
2. Positional encodings are added to the sentence embeddings to create the input for the encoder.
3. Multiple encoder blocks are stacked where each block has the following components-
    1. Multi-head self-attention followed by residual + layernorm
    2. Fully connected feed-forward network followed by residual + layernorm
4. Embeddings are created for the target sentence decoded so far (starts with an empty sentence `[]`).
5. Positional encodings are added to the embeddings of the target sentence.
6. Multiple decoder blocks are stacked where each block has the following components-
    3. Masked multi-head self-attention followed by residual + layernorm.
    4. Multi-head encoder-decoder attention followed by residual + layernorm.
    5. Fully connected feed-forward network followed by residual + layernorm.
7. Linear + softmax layer that finally gives the probabilities of the output words.


# Hyperparameters

All in all, we have the following hyperparameters and the values for each used by the authors of the paper:


1. D - Embedding dimension = 512
2. h - Number of attention heads = 8
3. n_enc - Number of encoder blocks = 6
4. n_dec - Number of decoder blocks = 6
5. D<sub>ff</sub> - Dimension of feed-forward layer = 2048
6. Dropout - Dropout rate for embeddings and output of each sub-layer = 0.1

This completes the description of the transformer model. All that is left are some details about the training process.

# Training

## Optimizer

The authors of the paper used Adam optimizer.


## Regularization

Dropout is applied to the output of each sub-layer before the residual + layernorm step. Dropout is also applied to the sum of embeddings and positional encodings for both encoder and decoder stacks.

# References

1. [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](https://jalammar.github.io/illustrated-transformer/)
3. [[1409.0473] Neural Machine Translation by Jointly Learning to Align and Translate (arxiv.org)](https://arxiv.org/abs/1409.0473)
4. [LSTM is dead. Long Live Transformers! - YouTube](https://www.youtube.com/watch?v=S27pHKBEp30)
5. [The Annotated Transformer (harvard.edu)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
6. [Adding vs. concatenating positional embeddings & Learned positional encodings - YouTube](https://www.youtube.com/watch?v=M2ToEXF6Olw&list=WL&index=8)