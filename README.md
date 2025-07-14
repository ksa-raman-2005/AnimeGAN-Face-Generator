# AnimeGAN-Face-Generator

A Deep Convolutional Generative Adversarial Network (DCGAN) built with PyTorch to generate high-quality anime-style faces. Trained using the popular Anime Face Dataset, this project demonstrates how deep learning models can be used for creative content generation.

---

## Project Overview

Generative Adversarial Networks (GANs) are a powerful deep learning approach for generating new data samples that resemble a given dataset. In this project, we use a DCGAN architecture to train on the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) and generate anime faces from random noise vectors.

This project:
- Implements both the Generator and Discriminator using PyTorch
- Trains the DCGAN from scratch
- Saves sample output images during training
- Exports the trained Generator model for reuse

---

## How It Works

The GAN architecture consists of two neural networks:

- **Generator (G)**: Takes random noise (`z`) as input and tries to generate realistic anime faces.
- **Discriminator (D)**: Tries to distinguish between real images from the dataset and fake images produced by the Generator.

Both networks are trained simultaneously in a minimax game:
- The Generator improves to fool the Discriminator.
- The Discriminator improves to distinguish fake from real images.

Over time, the Generator becomes good at generating realistic anime faces.

---

## Dataset

We use the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) from Kaggle.

You must download the dataset and place it in a suitable directory:
