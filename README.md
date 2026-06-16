# Super Resolution GAN: Deep Learning Image Upscaler

An implementation of a Generative Adversarial Network (GAN) designed to mathematically upscale and enhance low-resolution images, restoring high-fidelity details lost during compression or downsampling.

## Architectural Highlights

This project demonstrates advanced deep learning architecture, specifically focusing on feature preservation during spatial upscaling.

* **GAN-Based Upscaling:** Utilizes an adversarial training setup where a Generator synthesizes high-resolution images, and a Discriminator enforces perceptual realism.
* **Feature Extraction Pipeline:** Implements a deep Convolutional Neural Network (CNN) architecture to extract complex spatial hierarchies from low-resolution inputs.
* **Skip-Connection Layers:** Employs Residual Blocks (ResBlocks) with identity skip connections. This allows the network to bypass gradient vanishing during deep training and ensures low-level high-frequency image details are preserved through the upsampling process.

## Technology Stack
* **Framework:** PyTorch
* **Core Architectures:** Deep CNNs, GANs, PixelShuffle Upsampling
* **Language:** Python

## How to Run Inference

```bash
# Install dependencies
pip install -r requirements.txt

# Run the upscaler on a low-res target
python upscaler.py --input images/low_res_sample.jpg --output images/enhanced_sample.jpg
