---
layout: single
title: "Intro to Transformers"
twitter-image: /assets/posts/transformer/cards/transformer-intro_card.png
excerpt: "The Transformer is a deep learning model that uses “attention” to determine the relative importance of various parts of the input data."
header:
  teaser: /assets/posts/transformer/cards/transformer-intro_card.png
  overlay_image: /assets/posts/transformer/cards/transformer-intro_header.png
  overlay_filter: 0.5 # rgba(255, 0, 0, 0.5), linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 255, 255, 0.5))

date:   2021-07-25 20:30:00 +0530
last_modified_at: 2021-07-25 20:30:00 +0530
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

The transformer is a deep learning model used primarily for Natural Language Processing tasks, such as machine translation, but is slowly also finding use in tasks like computer vision with models like [Vision Transformer](https://arxiv.org/abs/2010.11929). Several of the biggest and best-performing language models like BERT and GPT-2/GPT-3 are transformer models or some variation of it. The architecture was introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762). In the coming sections we will follow the paper closely along with [this post from Jay Alammar](https://jalammar.github.io/illustrated-transformer/).

On a high level, transformers are encoder-decoder models, so the input strings (which can be of variable lengths) go through the encoder and get converted to a fixed-length representation in a latent space. This latent representation goes through the decoder and gives us the desired output, which might also be of variable length.

Let’s say we want to translate the sentence “_The transformer is a powerful and versatile machine learning model._” into German, which translates to “_Der Transformator ist ein leistungsstarkes und vielseitiges Modell für maschinelles Lernen._” (I don’t know German I used Google Translate for this). For our model to translate the source sentence into the target sentence, it will first need to encode it into some latent representation, this encoding step essentially extracts out the meaning of the sentence, which can then be used to convert it into a sentence in another language. In the second stage, the decoder takes this latent representation and generates the translated sentence one word at a time. The full architecture is shown below, we will see each of these components in detail in the coming sections.

{% include figure image_path="/assets/posts/transformer/full-architecture.png" alt="Transformer architecture" caption="The architecture of the transformer model. Source: [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)" %}


# Inputting data

The raw data we have is in the form of sentences, which are ordered lists of words. We convert them to an ordered list of tokens using the vocabulary we have for the language. These lists of numbers can be padded to some maximum length, and then batched to create BxN matrices, where B is the batch size and N is the maximum length of sentences. These tokens are then converted to embeddings of dimensions D, converting the sentence into a matrix of shape NxD, and thus making the batch a rank-3 tensor of shape BxNxD.  Working with tensors like this makes computation much more efficient, especially on GPUs which are made to do a lot of large matrix operations as efficiently as possible.


# Positional Encoding

Now that we have represented our sentences as matrices, we also need to encode the positional information of each word and combine it with the embeddings created in the previous step. Traditionally with LSTMs, the positional information of a word is implicitly present in the encoding as a result of sequentially processing the input sentence, but if we can store that information in the word embedding itself, it will allow us to process all the words of a sentence in parallel and still have information about each word’s position in the sentence.

For every position in the sentence, we need to create a representation that encodes the information about that position. Here, we have the option of using pre-defined encodings or learning them during training. If we create a P dimensional vector for encoding a given position in the sentence, then for a sentence of maximum length N, the positional encoding will be a matrix of size NxP. 

But how do we combine this information with the word embeddings to have a unified representation that has information about both the word as well as its position? One way is to concatenate the positional embedding with the word embedding, making the final representation of each word a (D+P) dimensional vector, and thus the representation of a complete sentence will be a matrix of shape Nx(D+P). We can see this increases the complexity of our model and thus increase the memory and compute requirement for training. 

Another way is to simply add the positional encodings to the word embeddings. This might not seem like a good idea at first but the authors of the paper found this method to work very well and it also decreases the complexity of the model. For more detailed discussion check out [this video](https://www.youtube.com/watch?v=M2ToEXF6Olw&list=WL&index=8). To be able to add the two vectors, their dimensionality needs to be the same, i.e., P=D. This also saves us the trouble of having to tune another hyperparameter during training. In terms of whether we should use pre-made or learned positional encodings, the authors tried both and found that there was no significant difference in performance between the two methods, and finally decided to use the pre-defined encodings because those would likely generalize better to sentences longer than the ones found in the training data.

The formula used for positional encoding is as follows:

$$ PE_{(pos, 2i)} = sin(\frac{pos}{10000^{2i/D}}) $$

$$ PE_{(pos, 2i+1)} = cos(\frac{pos}{10000^{2i/D}}) $$

The positional encodings are generated by interleaving sine and cosine functions of different frequencies for different dimensions. This representation does not look very intuitive at first glance, but it looks somewhat like a Fourier Transform, so we can think of it as bringing the information of time domain - i.e., position in the sentence, into the space domain - i.e., word embedding.

The authors of the paper explain the choice of this function as follows:

> We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset _k_, _PE<sub>pos+k</sub>_ can be represented as a linear function of _PE<sub>pos</sub>_.

{% include figure image_path="/assets/posts/transformer/pos-encode.png" alt="Positional encodings" caption="Example of positional encodings for embedding size = 128 and maximum tokens in a sentence = 20." %}

So now, we have word embeddings and positional encodings, and after adding these two we arrive at the final representation of our batch of sentences with the shape BxNxD that are ready to be input into the encoder and start the translation process.

# References

1. [[1706.03762] Attention Is All You Need (arxiv.org)](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer – Jay Alammar – Visualizing machine learning one concept at a time. (jalammar.github.io)](https://jalammar.github.io/illustrated-transformer/)
3. [[1409.0473] Neural Machine Translation by Jointly Learning to Align and Translate (arxiv.org)](https://arxiv.org/abs/1409.0473)
4. [LSTM is dead. Long Live Transformers! - YouTube](https://www.youtube.com/watch?v=S27pHKBEp30)
5. [The Annotated Transformer (harvard.edu)](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
6. [Adding vs. concatenating positional embeddings & Learned positional encodings - YouTube](https://www.youtube.com/watch?v=M2ToEXF6Olw&list=WL&index=8)