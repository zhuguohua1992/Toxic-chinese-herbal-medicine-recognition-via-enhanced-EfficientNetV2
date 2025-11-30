# Toxic Chinese Herbal Medicine Recognition in Real-World Images via Multi-Scale and Attention-Enhanced EfficientNetV2

This repository provides the official PyTorch implementation of our work **"Toxic Chinese Herbal Medicine Recognition in Real-World Images via Multi-Scale and Attention-Enhanced EfficientNetV2"**.

The project focuses on **fine-grained recognition of 47 toxic Chinese herbal medicines** in **real-world images with complex backgrounds, small target regions, and challenging lighting conditions**.  
Building upon EfficientNetV2, we introduce:

- a **Multi-Scale Feature Fusion (MSFF)** module to better capture multi-level semantic and structural information;  
- a **Convolutional Block Attention Module (CBAM)** to adaptively emphasize discriminative herbal regions while suppressing cluttered backgrounds.

Under the same training settings as ResNet, ResNeXt, and vanilla EfficientNet baselines, the proposed MSFF + CBAM enhanced EfficientNetV2 achieves **consistently higher Top-1 accuracy, mean precision, mean recall, and macro F1-score**, especially on a curated **Small-Target & Complex-Background Challenge Set**.

This codebase includes:
- model definitions (EfficientNetV2 + MSFF + CBAM),
- training and evaluation scripts,
- configuration examples for reproducing the main experimental results reported in the paper.

If you use this repository in your research, please consider citing our work.

