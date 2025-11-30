# Toxic Chinese Herbal Medicine Recognition via Multi-Scale and Attention-Enhanced EfficientNetV2

This repository contains the official PyTorch implementation for the paper:

**"Toxic Chinese herbal medicine recognition in real-world images via multi-scale and attention-enhanced EfficientNetV2"**

## 📌 Overview
This project introduces a Multi-Scale Feature Fusion (MSFF) module and a CBAM attention block into the EfficientNetV2 backbone to improve recognition accuracy of toxic Chinese herbal medicines under real-world visual environments with complex backgrounds and small target regions.

## ✨ Key Features
- Multi-Scale Feature Fusion for enhanced representation of small objects
- CBAM attention to suppress background noise and focus on herbal regions
- Strong performance improvement over ResNet, ResNeXt, EfficientNet, and Transformer baselines
- Robust performance on challenging real-scene images

## 🚀 Training
```example
python tools/train.py models/efficientnetv2/Enhanced-efficientnetv2.py
