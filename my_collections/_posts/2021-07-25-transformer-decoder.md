---
layout: single
title: "Transformer Decoder"
twitter-image: /assets/posts/transformer/cards/transformer-decoder_card.png
excerpt: "The decoder in a transformer model uses stacks of decoder blocks made up of self-attention, encoder-decoder attention, and feed-forward layers for generating the output sequence."
header:
  teaser: /assets/posts/transformer/cards/transformer-decoder_card.png
  overlay_image: /assets/posts/transformer/cards/transformer-decoder_header.png
  overlay_filter: 0.5 # rgba(255, 0, 0, 0.5), linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 255, 255, 0.5))

date:   2021-07-25 20:45:00 +0530
last_modified_at: 2021-07-25 20:45:00 +0530
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

The decoder works very similarly to the encoder, with a few important differences. Like the encoder, the decoder is also a stack of multiple decoder blocks with the same architecture but different weights. However, in a decoder block, there are two attention sub-layers before the feed-forward sub-layer.


# Masked Self-Attention

The first attention sublayer is also a multi-head self-attention sublayer, but this time, a word is only allowed to attend to itself and the words before it, and no words that come after it. This is done by masking the future words, i.e., by replacing the values of the cells (i, j) where (j > i) in the attention matrix ($ Q \times K^{T} $) with `-infinity` before the softmax step. This means that after the softmax step, the attention value of future words will become zero.

This is done because while creating the translated sentence, we start with an empty sentence and create the sentence one word at a time until a special end-of-sentence character is generated or the maximum length of the output sentence is reached.


# Encoder-Decoder Attention

The second attention sub-layer is not self-attention, rather, it creates the Query matrix using the output from the previous masked self-attention layer and Key and Value matrices using the output from the encoder, i.e., output from the last encoder block in the encoder stack. If the maximum sentence length of the output sentence is M, then a sentence is represented as an MxD matrix, which is also the shape of the output from the masked self-attention sub-layer. Using this, we get the Query matrix of shape MxD<sub>K</sub> (remember D<sub>Q</sub>=D<sub>K</sub>), while the K and V matrices come from the encoder and have shapes NxD<sub>K</sub> and NxD<sub>V</sub> respectively. This gives us the attention matrix of shape MxN. Finally, this process gives us the Value matrix of shape MxD<sub>V</sub>. Concatenating the results of multiple attention heads and multiplying with the weights matrix like in the previous cases gives us the final output of shape MxD. M and N can be different if required but M=N usually also works well. The embedding dimension (D) must be the same for both the source and target language.

{% include figure image_path="/assets/posts/transformer/encoder-decoder-attention-example.png" alt="Encoder-decoder attention" caption="Illustration of encoder-decoder attention between source and translated sentence. Notice how the model attends to the relevant words even when their positions in source and target sentences is different. Source: [[1409.0473] Neural Machine Translation by Jointly Learning to Align and Translate (arxiv.org)](https://arxiv.org/abs/1409.0473)" %}

# Feed-Forward NN and Residual + LayerNorm

Once again, like the encoder, we have a feed-forward neural network with an intermediate layer with shape D<sub>ff</sub> and ReLU activation and a final layer of shape D. Similar to the encoder, all these sub-layers have residual and layer norm components, so the complete output of a decoder block becomes:

$$ DecoderBlock(X, Y) = LayerNorm(Z_2 + FeedForward(Z_2)) $$

Where,

$$ Z_2 = LayerNorm(Z_1 + EncoderDecoderAttention(Z_1, Y)) $$

$$ Z_1 = LayerNorm(X + MaskedSelfAttention(X)) $$

$$ X = Target \: sentence \: decoded \: so \: far $$

$$ Y = Output \: of \: the \: encoder $$

{% include figure image_path="/assets/posts/transformer/decoder-block.png" alt="Decoder block" caption="Block diagram of decoder block. Source: [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)" %}

Once again, the decoder consists of multiple decoder blocks stacked on top of one another.


# Final Result

After multiple decoder blocks as described above, we finally get to the linear + softmax layer to generate the output tokens. The linear layer takes us from the D dimensional embedding space to a V dimensional space, where V is the vocabulary size of the target language. Each dimension in this space corresponds to one word in the target language. We apply softmax to this result which converts the output into probabilities of different words. Now, we can get the generated token either by taking the word with the highest probability or by sampling randomly using the probability values given by the softmax layer.

# References

1. [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](https://jalammar.github.io/illustrated-transformer/)
3. [[1409.0473] Neural Machine Translation by Jointly Learning to Align and Translate (arxiv.org)](https://arxiv.org/abs/1409.0473)
4. [LSTM is dead. Long Live Transformers! - YouTube](https://www.youtube.com/watch?v=S27pHKBEp30)
5. [The Annotated Transformer (harvard.edu)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
6. [Adding vs. concatenating positional embeddings & Learned positional encodings - YouTube](https://www.youtube.com/watch?v=M2ToEXF6Olw&list=WL&index=8)