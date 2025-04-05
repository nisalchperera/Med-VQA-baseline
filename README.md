# Med-VQA: Medical Visual Question Answering

## Introduction

Imagine being able to chat with medical images and get accurate responses to medical queries. That's exactly what Med-VQA aims to achieve! Med-VQA is a Medical Visual Question Answering (VQA) system that takes a medical image and a related question as input and predicts a relevant answer.

This problem can be divided into the following key components:

- Extracting features from medical images

- Embedding and processing the textual question

- Fusing both text and image features

- Predicting the final answer

## Baseline Model Architecture

The baseline Med-VQA model follows this pipeline:

**Image Feature Extraction**: Uses VGG16 or ResNet to extract key features from medical images.

**Text Embedding**: Embeds the medical question using BioBERT, a domain-specific BERT variant trained on biomedical text.

**Multimodal Fusion**: Combines both visual and textual features to generate a meaningful representation by concatenating the image and text features

**Answer Prediction**: Uses a fully connected neural network to predict the answer based on the fused features.

# Run the Baseline model
- step 1: Download the VQA-RAD dataset and keep in the data folder. If data folder is not present make a data folder and then keep tthe vqa data folder inside
- step 2: pip install -r requirements.txt
- setp 3: python train.py