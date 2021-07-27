---
layout: single
title: "Transformer Encoder"
twitter-image: /assets/posts/transformer/cards/transformer-encoder_card.png
excerpt: "The encoder in a transformer model uses stacks of encoder blocks made up of self-attention and feed-forward layers for encoding the input sequence."
header:
  teaser: /assets/posts/transformer/cards/transformer-encoder_card.png
  overlay_image: /assets/posts/transformer/cards/transformer-encoder_header.png
  overlay_filter: 0.5 # rgba(255, 0, 0, 0.5), linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 255, 255, 0.5))

date:   2021-07-25 20:40:00 +0530
last_modified_at: 2021-07-25 20:40:00 +0530
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

# Intro

The encoder consists of several encoder blocks, which have the same architecture but do not share weights. Think of them as the hidden layers in a dense neural network, or like convolution + max-pooling blocks in a ConvNet. Each additional encoder block adds the ability for the model to learn higher-level representations of the text. The encoder block contains a self-attention sub-layer and a feed-forward neural network sub-layer. There are residual connections and layer normalization steps after both sub-layers. Let’s see each one of them in little more detail.


# Self-Attention

We have seen how attention works in the previous section. Self-attention is when the Query, Key, and Value matrices all come from the same sentence. So while encoding the input sentence, we are calculating the self-attention, or how much attention to pay to the words of the input sentence while encoding the word. This is slightly different from the implementation of attention in the decoder layer which we will see later.


# Feed-Forward Neural Network

After getting the output from the attention layer, it goes through a fully connected feed-forward neural network. Each word goes through this network separately and identically. There are two layers in this network with a ReLU activation after the first layer. The first layer has an output of shape D<sub>ff</sub> which is another tunable hyperparameter of this model. The second layer has an output of shape D, so that the final output of this sub-layer has the same shape as the input.


# Residual + LayerNorm

A residual connection means to simply add the input to the output of a given function. 

{% include figure image_path="/assets/posts/transformer/residual.png" alt="Residual connection illustration" caption="Illustration of residual connection." %}

After getting the output of the sublayers, we apply a residual connection and then apply layer normalization to the output of the residual.


# Final result

Putting all of these together, the output of the encoder block looks something like this:

$$ EncoderBlock(X) = LayerNorm(Z + FeedForward(Z)) $$

Where,

$$ Z = LayerNorm(X + MultiHeadAttention(X)) $$

$$ X = Input \: sentence $$

{% include figure image_path="/assets/posts/transformer/encoder-block.png" alt="Encoder block" caption="Block diagram of the encoder block. Source: [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)" %}

The complete encoder is a stack of multiple encoder blocks like this, with each encoder block giving an output of the same shape as its input, but internally the matrix is transforming from containing the embeddings of individual words to containing a high-level abstract meaning of the whole sentence.

# References

1. [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](https://jalammar.github.io/illustrated-transformer/)
3. [[1409.0473] Neural Machine Translation by Jointly Learning to Align and Translate (arxiv.org)](https://arxiv.org/abs/1409.0473)
4. [LSTM is dead. Long Live Transformers! - YouTube](https://www.youtube.com/watch?v=S27pHKBEp30)
5. [The Annotated Transformer (harvard.edu)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
6. [Adding vs. concatenating positional embeddings & Learned positional encodings - YouTube](https://www.youtube.com/watch?v=M2ToEXF6Olw&list=WL&index=8)