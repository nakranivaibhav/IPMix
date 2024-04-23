# IPMix: Label-Preserving Data Augmentation for Robust Classifiers

This repository contains a PyTorch implementation of the IPMix data augmentation method proposed in the paper "IPMix: Label-Preserving Data Augmentation Method for Training Robust Classifiers." IPMix is a novel approach that integrates three levels of data augmentation techniques (image-level, patch-level, and pixel-level) into a single framework to improve the robustness and accuracy of deep neural network classifiers.

## What is IPMix?

IPMix is a data augmentation technique that combines different augmentation methods to generate diverse and complex training samples. It mixes the original input image with synthetic images (e.g., fractals) at various levels, including image-level transformations (like brightness and sharpness adjustments), patch-level replacements, and pixel-level mixing. This approach increases the structural complexity and diversity of the training data, leading to improved model robustness against corruptions, adversarial attacks, and other distribution shifts, without compromising accuracy on clean data.

## Features

- **Label-preserving**: IPMix ensures that the generated augmented images retain the original label, avoiding potential issues like manifold intrusion.
- **Multi-level augmentation**: IPMix integrates three levels of data augmentation techniques (image-level, patch-level, and pixel-level) into a coherent framework.
- **Increased diversity**: By mixing the input image with synthetic images (fractals) and applying various transformations, IPMix generates highly diverse training samples, improving model robustness.
- **Limited computational overhead**: IPMix does not require expensive searching for optimal augmentation policies, making it computationally efficient.
