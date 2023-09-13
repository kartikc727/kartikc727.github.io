---
title: "Adversarial Examples"
excerpt: "Creating adversarial examples for image classifiers using fast gradient sign method"
collection: portfolio
date: 2021-07-01 20:30:00 +0530
last_modified_at: 2021-12-24 08:30:00 +0530
---
[![Open In Colab][colab-badge]][colab-notebook] [![Github forks][gh-fork-shield]][github-repo]

Adversarial examples are inputs created with the intention of getting an incorrect output from a neural network.
They can be used to test the robustness of models to small perturbations or for malicious purposes by bad actors.

## How do they work

I am planning on creating a detailed post on adversarial examples, but for now, we will look at one particular type of
adversarial attack called the Fast Gradient Sign Method or FGSM. We will follow [this tutorial][1] from TensorFlow.

Let's say we have a well-trained network that correctly predicts most of the images from the CIFAR-10 dataset.
I am given an image of a bird, which is correctly classified as a bird by the model. Now, I want to make small perturbations
to the image such that there is little to no visual change to the image but now the model starts misclassifying it to, let's
say, a frog.

{% include figure 
image_path="images/assets/projects/adversarial/adversarial_attack.png"
alt="Adversarial example" 
caption="Changing the image in a way that is imperceptible to humans can still lead to change in the prediction by neural networks"
 %}

These perturbations are not random noise but carefully generated in a way to make the image be incorrectly classified by our
model. To generate this "noise" image, we give the original image as an input to the model and look at the gradients of the
loss **with respect to the input image**. Then, we take the sign of these gradients. This will tell us the direction in which
we need to change our input (the image of the bird) to get the largest amount of change in the prediction. 

This method requires access to the model architecture and weights in order to calculate the gradients, making it a white
box attack. There are other methods that do not require the weights and only use the predict function to create adversarial
examples. Even physical objects can be used in many cases to create adversarial examples.

## Dataset

In this example, we will look at a simple image classification network trained on the CIFAR-10 dataset. I have
chosen this dataset for convenience since it will allow me to build a smaller network that is easy to train and
experiment with using only the free resources from Google Colab.

## Training the model

The first part is relatively simple. We train an image classification model in TensorFlow - A simple ConvNet that we will
later attack with adversarial examples.

## Creating adversarial examples using FGSM

Now let's take a couple of images that our model currently classifies correctly and manipulate them so that they get 
classified incorrectly.

```python
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    '''
    Gets the signs of the gradients for the input image
    '''
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad
```
For the fast gradient sign method, we just need to calculate the gradients of the output w.r.t the input image, and then
take the sign of those gradients along each dimension. This gives us the perturbation that we need to apply to the image
to change the output while the image visually remains the same.

```python
new_image = image + eps * signed_grad
```
We can see why this method is called the 'fast' gradient sign method. We just need to calculate the gradients once and then
add the resulting gradient signs in whatever proportion we want.

{% include figure 
image_path="images/assets/projects/adversarial/fgsm_example.png"
alt="Example of images" 
caption="Examples where images are gradually perturbed using the gradients and how the perturbations affect the model predictions."
 %}

## Why does this work?

We are exploiting the fact that neural network image classifiers don't work the same way as our brains. Tiny changes in the
illumination or color of a few pixels are not perceptible to humans but these artificial neural nets can see them just fine.
Unless we specifically instruct them to ignore these tiny changes, if they help them get their training loss down, the model
will use them and pay attention to them.

## References

1. [Adversarial example using FGSM - TensorFlow Core][1]
2. [Adversarial Examples - Interpretable Machine Learning (christophm.github.io)][2]
3. [Physical Adversarial Examples Against Deep Neural Networks – The Berkeley Artificial Intelligence Research Blog][3]


<!-- Links -->
[colab-badge]: <https://colab.research.google.com/assets/colab-badge.svg>
[colab-notebook]: <https://colab.research.google.com/github/kartik727/ml-projects/blob/master/adversarial-examples/Adversarial_examples.ipynb> "Colab notebook"
[gh-fork-shield]: <https://img.shields.io/github/forks/kartik727/ml-projects.svg?style=social&label=Fork&maxAge=2592000>
[github-repo]: <https://github.com/kartik727/ml-projects/tree/master/adversarial-examples> "Github repository"
[1]: <https://www.tensorflow.org/tutorials/generative/adversarial_fgsm> "Adversarial example using FGSM - TensorFlow Core"
[2]: <https://christophm.github.io/interpretable-ml-book/adversarial.html> "Adversarial Examples - Interpretable Machine Learning (christophm.github.io)"
[3]: <https://bair.berkeley.edu/blog/2017/12/30/yolo-attack/> "Physical Adversarial Examples Against Deep Neural Networks – The Berkeley Artificial Intelligence Research Blog"
