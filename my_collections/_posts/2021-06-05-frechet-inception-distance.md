---
layout: single
title: "Frechet Inception Distance"
twitter-image: /assets/images/default-teaser.jpg
excerpt: "Frechet Inception Distance is a metric to measure the quality of images generated by GANs and other generative models."
header:
  teaser: /assets/images/default-teaser.jpg
  overlay_image: /assets/images/sample-img-wide.jpg
  overlay_filter: 0.5 # rgba(255, 0, 0, 0.5), linear-gradient(rgba(255, 0, 0, 0.5), rgba(0, 255, 255, 0.5))
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
  
date:   2021-06-05 20:30:00 +0530
last_modified_at: 2021-06-08 20:30:00 +0530
categories: ML GAN
published: true

# table
toc: true
# toc_label: "Label goes here"
# toc_icon: "<some font awesome icon>"
toc_sticky: true

# sidebar
sidebar:
  - title: "Generative Adversarial Networks"
    image: /assets/images/ml/gan-sidebar.png
    image_alt: "Images generated by GANs"
    text: "GANs are generative models that can synthesize realistic images similar to the ones they are trained on."
  - nav: gans

fid-scores:
  - url: /assets/posts/fid/1.png
    image_path: /assets/posts/fid/1.png
    alt: "example image 1"
    title: "Gaussian Noise"
  - url: /assets/posts/fid/2.png
    image_path: /assets/posts/fid/2.png
    alt: "example image 2"
    title: "Gaussian Blur"
  - url: /assets/posts/fid/3.png
    image_path: /assets/posts/fid/3.png
    alt: "example image 3"
    title: "Implanted Black Rectangles"
  - url: /assets/posts/fid/4.png
    image_path: /assets/posts/fid/4.png
    alt: "example image 4"
    title: "Swirled Images"
  - url: /assets/posts/fid/5.png
    image_path: /assets/posts/fid/5.png
    alt: "example image 5"
    title: "Salt and Pepper Noise"
  - url: /assets/posts/fid/6.png
    image_path: /assets/posts/fid/6.png
    alt: "example image 6"
    title: "CelebA Dataset Contaminated by ImageNet Images"
---

# TL;DR

FID is a metric used to evaluate the quality of images generated from GANs, it uses the activations from the final pooling layer of the Inception V3 model to represent the images, and calculates the distance or dissimilarity of the sets of real and generated sets using the following formula:

$$ FID = ||\mu_1 - \mu_2||^2 + tr(\Sigma_1 + \Sigma_2 - 2\sqrt{\Sigma_1\Sigma_2})  $$

Where  $\mu$ is the mean vector and $\Sigma$ is the covariance matrix. For a more detailed explanation, keep reading.


# Introduction
The basic idea behind Frechet Inception Distance or FID is that in order for the fake images to be indistinguishable from the real ones, they should come from the same distribution of images that we would consider real.

Let’s say we have a bunch of images that we generated using our GAN, or any other method of generating images, and we want to compare them to a set of real images to get a sense of how well our model did. One way to do this could be to manually look at some example images to get a general idea of what our model is producing.

While this is easy to do and gives us a feel of how good our model is, one might also want a more objective way to measure GAN performance to compare different models and to report their findings to others. 

One such metric is the Frechet Inception Distance. FID is the distance between the distributions of real and fake images. The distance in this context is a measure of dissimilarity between the two distributions. To calculate the distance, we take the Inception V3 model, but instead of using the final classifications from the model (like in Inception Score), we use it as a feature extractor and take the activations from the last pooling layer of the model before the fully connected layers and see how the images are distributed in that space.


# Calculating the distance

Here, we make the assumption that the distributions take the form of multivariate normal distributions which simplifies our calculation and makes computation easy.

A multivariate normal distribution can be parameterized by its mean vector $(\mu)$ and covariance matrix $(\Sigma)$.

The formula for calculating the distance is:

$$ FID = ||\mu_1 - \mu_2||^2 + tr(\Sigma_1 + \Sigma_2 - 2\sqrt{\Sigma_1\Sigma_2})  $$

Here, tr(X) is the trace of the matrix X, and \|\|X\|\| is the norm of X. Distributions 1 and 2 are of the real and generated images respectively. Also, note that $ \sqrt{\Sigma_1\Sigma_2} $ is the square root of the resulting matrix $ \Sigma_1\Sigma_2 $ and not element-wise square root.

This has the dimensions of distance squared and gives us a measure of how far apart the two distributions are. The closer the two distances, the lower the FED and the better your generated images. We should use a large number of real and generated images for this calculation to reduce noise and get meaningful results from the calculation.

You can check out [this blog post by Jason Brownlee](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/) for the Python implementation of the FID.

{% include gallery id="fid-scores" caption="Examples of how FID score changes due to image distortions. Source: [GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium](https://arxiv.org/abs/1706.08500)" %}


# Advantages



1. This metric is easy to calculate once the images are generated.
2. It is based on the Inception model trained on the ImageNet dataset that covers a large variety of classes, so it is applicable in lots of use cases.
3. Unlike the Inception Score, it compares the distributions of the real and fake images, while the Inception Score only looks at the generated images.


# Limitations



1. Requires a large amount of data (both real and fake images) to give good results.
2. Only good on tasks that are a subset of ImageNet, or close enough that the embeddings make sense.
3. Does not capture everything about the distributions, just the first two moments.
4. The metric is biased and depends on the number of samples, which limits its usefulness as a good metric to benchmark different GANs.


# References

[Build Better Generative Adversarial Networks (GANs) \| Coursera](https://www.coursera.org/learn/build-better-generative-adversarial-networks-gans/home/welcome)

[GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (arxiv.org)](https://arxiv.org/abs/1706.08500)

[How to Implement the Frechet Inception Distance (FID) for Evaluating GANs (machinelearningmastery.com)](https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/)

[Fréchet inception distance - Wikipedia](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)




