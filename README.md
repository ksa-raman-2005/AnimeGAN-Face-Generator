# AnimeGAN-Face-Generator

A Deep Convolutional Generative Adversarial Network (DCGAN) built with PyTorch to generate high-quality anime-style faces. Trained using the popular Anime Face Dataset, this project demonstrates how deep learning models can be used for creative content generation.

---

## Project Overview

Generative Adversarial Networks (GANs) are a powerful deep learning approach for generating new data samples that resemble a given dataset. In this project, we use a DCGAN architecture to train on the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) and generate anime faces from random noise vectors.

This project:

* Implements both the Generator and Discriminator using PyTorch
* Trains the DCGAN from scratch
* Saves sample output images during training
* Exports the trained Generator model for reuse

---

## How It Works

The GAN architecture consists of two neural networks:

* **Generator (G)**: Takes random noise (`z`) as input and tries to generate realistic anime faces.
* **Discriminator (D)**: Tries to distinguish between real images from the dataset and fake images produced by the Generator.

Both networks are trained simultaneously in a minimax game:

* The Generator improves to fool the Discriminator.
* The Discriminator improves to distinguish fake from real images.

Over time, the Generator becomes good at generating realistic anime faces.

---

## Dataset

We use the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) from Kaggle.

You must download the dataset and place it in a suitable directory:

```bash
/your_path/.cache/kagglehub/datasets/splcher/animefacedataset/versions/3
```

---

## Model Architecture

### Generator

The Generator network transforms a random noise vector of shape `(100, 1, 1)` into a `64x64` RGB image through a series of transposed convolutional layers.

**Layers:**

* ConvTranspose2d -> BatchNorm -> ReLU (x4)
* Final Layer: ConvTranspose2d -> Tanh (for \[-1, 1] output)

### Discriminator

The Discriminator network takes a `64x64` image and outputs a single probability score (real/fake).

**Layers:**

* Conv2d -> LeakyReLU -> BatchNorm (x4)
* Final Layer: Conv2d -> Sigmoid

---

## Training Pipeline

1. **Preprocessing**:

   * Resize all images to `64x64`
   * Normalize to \[-1, 1] range

2. **Safe Dataset**:

   * Skips broken or corrupt images that don't match dimensions

3. **Weight Initialization**:

   * Normal distribution (mean=0, std=0.02) for stable training

4. **Training Loop**:

   * For each batch:

     * Train Discriminator on real and fake images
     * Train Generator to fool the Discriminator
   * Save generated samples at the end of each epoch

5. **Saving Output**:

   * Saves output images in `/samples`
   * Saves final Generator model as `anime_generator.pth`

---

## Installation & Requirements

### Dependencies

* Python 3.8+
* PyTorch
* torchvision
* matplotlib
* tqdm

### Installation

```bash
pip install torch torchvision matplotlib tqdm
```

---

## Running the Project

```bash
python AnimeGAN_Main.py
```

Make sure the dataset is properly placed and the path in the script is correct.

---

## Output Samples

Generated samples will be stored in the `/samples` folder. Example output:

---

## Saving the Generator

After training, the Generator is saved:

```bash
anime_generator.pth
```

You can reload it to generate new anime faces later using:

```python
netG.load_state_dict(torch.load("anime_generator.pth"))
```

---

## Future Improvements

* Train on larger datasets
* Add a UI for real-time face generation
* Fine-tune for specific art styles

---

## License

This project is for educational purposes only.

---

## Credits

* Kaggle Anime Dataset: [splcher/animefacedataset](https://www.kaggle.com/datasets/splcher/animefacedataset)
* DCGAN architecture inspired by PyTorch's official DCGAN tutorial
