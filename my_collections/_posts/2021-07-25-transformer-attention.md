---
layout: single
title: "Attention"
twitter-image: /assets/posts/transformer/cards/transformer-attention_card.png
excerpt: "Attention is a mechanism that allows sequence models to attend to the relevant parts of the input data by mimicking how humans can selectively concentrate on a few things."
header:
  teaser: /assets/posts/transformer/cards/transformer-attention_card.png
  overlay_image: /assets/posts/transformer/cards/transformer-attention_header.png
  overlay_filter: 0.5 # rgba(255, 0, 0, 0.5), linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 255, 255, 0.5))

date:   2021-07-25 20:35:00 +0530
last_modified_at: 2021-07-25 20:35:00 +0530
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

Unlike classical RNN models like LSTMs, attention works in a way that can be easily parallelized and processed efficiently on GPUs, significantly improving scalability for larger language models.

Attention is at the core of the transformer model and it’s what makes the model perform as well as it does.  This is evident in the fact that the paper introducing the transformer architecture is titled “Attention is all you need”. 

To understand what attention is, let’s go back to our example - we want to encode our sentence “_The transformer is a powerful and versatile machine learning model_”. While encoding each word, we need to look at other words in the sentence to gather the complete meaning of that word. When encoding the word “_powerful_”, we also want to know what is the thing we are calling powerful - “_The transformer_”. Attention is the mechanism that the transformer model uses to gain this ability of understanding relationships between different words of a sentence.

The way attention works is that for each word, we create 3 vectors: Query, Key, and Value vectors. These vectors are created by using three trainable matrics **W<sup>Q</sup>**, **W<sup>K</sup>**, and **W<sup>V</sup>** of shapes DxD<sub>Q</sub>, DxD<sub>K</sub>, and DxD<sub>V</sub> respectively. For a sentence of N words, this translates to doing a matrix multiplication between the NxD matrix containing our sentence and these three matrices, giving us the **Q**, **K**, and **V** matrices of respective dimensionality NxD<sub>Q</sub>, NxD<sub>K</sub>, and NxD<sub>V</sub>. 

$$ X \times W^{Q} = Q $$

$$ X \times W^{K} = K $$

$$ X \times W^{V} = V $$

Where,

$$ X = Input \: sentence $$

By multiplying the Query matrix and the transpose of the Key matrix, we get the NxN attention matrix which tells the model how much “attention” to pay to each word while encoding a particular word. We can see that this matrix multiplication means the values D<sub>Q</sub> and D<sub>K</sub> must be the same. While D<sub>V</sub> doesn’t have such a constraint, it is also sometimes kept the same as the other two.

$$ A = Q \times K^{T} $$


# Dot-Product Attention

Let’s look at these matrices in a little bit more detail. In the attention matrix, the value of the (i, j)<sup>th</sup> cell is the dot product of the i<sup>th</sup> row in the **Q** matrix and j<sup>th</sup> row in the **K** matrix. Since the dot product gives us a measure of similarity between two vectors, we can think of the value of the (i, j)<sup>th</sup> cell as the relevance of word _j_ while encoding word _i_ in the sentence.

Finally, softmax is applied to this matrix to normalize the attention values such that the “relevance scores” of all the words used for encoding a particular word add up to 1. In practice, the matrix is also divided by the square root of the size of key vectors ($ \sqrt{D_K} $) before the softmax step , this scaling helps to prevent the values from going into regions of the softmax function with extremely small gradients. This attention matrix is finally multiplied by the Value matrix so that now each word’s encoding is a sum of encodings of the input words scaled by their relevance. The final formula for dot-product attention thus becomes:

$$ Attention(Q, K, V) = softmax(\frac{Q \times K^T}{\sqrt{D_K}}) \times V $$

{% include figure image_path="/assets/posts/transformer/scaled-dot-product-attention.png" alt="Scaled dot-product attention" caption="Block diagram for scaled dot-product attention. Source: [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)" %}

While implementing this in batch settings, with the input of shape BxNxD, we can use something like `torch.bmm` function in PyTorch, which does the matrix multiplications independently for the B sentences in our batch and gives us the final output of the shape BxNxD<sub>V</sub>

{% include figure image_path="/assets/posts/transformer/self-attention-example.png" alt="Attention illustration" caption="Relationships captured by the attention mechanism. Source: [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)" %}

# Multi-head attention

We saw how attention calculates the “relevance scores” of different words and thus how much attention to pay to them while encoding our sentence, but there can be lots of different aspects of what relevant means. It could relate to the grammatical structure of the sentence, or the gender of different words, or many other things. To capture these different aspects of the relationship between different words, we can repeat the steps of calculating attention multiple times, with different **W<sup>Q</sup>**, **W<sup>K</sup>**, and **W<sup>V</sup>** matrices that are all randomly initialized at the start of the training process. Then during training, these different “heads” of attention can hopefully learn these different aspects of attention.

{% include figure image_path="/assets/posts/transformer/multi-head-attention.png" alt="Multi-head attention" caption="Block diagram for scaled multi-head attention. Source: [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)" %}

Think of these different attention heads like different channels in a convolution layer of ConvNets, with each channel having its own kernel that encodes a different aspect of an image.

So now, if we have _h_ attention heads, we get _h_ outputs of attention with shape NxD<sub>V</sub>. To combine them, we simply concatenate them giving us an output of shape Nx(hD<sub>V</sub>). Finally, we multiply this with a weights matrix (**W<sup>O</sup>**) of shape (hD<sub>V</sub>)xD which projects the attention values in the space of the word embeddings, giving us the output of multi-head attention in the shape NxD, which is what the downstream layers are expecting.

{% include figure image_path="/assets/posts/transformer/two-heads-self-attention-example.png" alt="Multi-head attention example" caption="Example of how different heads in multi-head attention might capture different aspects of the relationships between words. Source: [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)" %}

$$ MultiHeadAttention(Q, K, V) = Concat(Attention(Q_i, K_i, V_i)) \times W^O $$

Once again, implementing this in a batch setting we will use batch matrix multiplication giving us the output of shape BxNxD.

# References

1. [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](https://jalammar.github.io/illustrated-transformer/)
3. [[1409.0473] Neural Machine Translation by Jointly Learning to Align and Translate (arxiv.org)](https://arxiv.org/abs/1409.0473)
4. [LSTM is dead. Long Live Transformers! - YouTube](https://www.youtube.com/watch?v=S27pHKBEp30)
5. [The Annotated Transformer (harvard.edu)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
6. [Adding vs. concatenating positional embeddings & Learned positional encodings - YouTube](https://www.youtube.com/watch?v=M2ToEXF6Olw&list=WL&index=8)