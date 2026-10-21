---
title: SmartVision CIFAR-10 Classifier
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# SmartVision — CIFAR-10 Image Classifier

SmartVision is an image classification web application
developed using PyTorch and Gradio.

## Project Overview

The application uses a TinyVGG convolutional neural
network to classify images into 10 CIFAR-10 categories.

## Supported Categories

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Model Architecture

- Framework: PyTorch
- Architecture: TinyVGG
- Input: RGB images resized to 32 x 32 pixels
- Output: 10 class probabilities
- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Training epochs: 10

## Model Evaluation

- Training images: 45,000
- Validation images: 5,000
- Test images: 10,000
- Test accuracy: 57.44%
- Weighted precision: 57.40%
- Weighted recall: 57.44%
- Weighted F1-score: 57.05%

## How to Use

1. Upload an image.
2. Click Submit.
3. View the predicted object classes.
4. View the prediction time in seconds.

## Limitations

The model was trained on low-resolution CIFAR-10 images.
Predictions on arbitrary photographs may be less reliable.
