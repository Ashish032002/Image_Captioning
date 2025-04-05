# Medical Image Captioning Using Generative Pretrained Transformers

## Overview
This repository contains the implementation of a novel approach for automated radiology reporting, specifically designed for chest X-ray captioning. The model integrates the Show-Attend-Tell (SAT) framework with Generative Pretrained Transformer (GPT) to generate detailed radiology reports from medical images.

## Authors
- Ashish Singh, Vellore Institute of Technology, Andhra Pradesh (VIT-AP University)
- Deepshika Mishra, Vellore Institute of Technology, Andhra Pradesh (VIT-AP University)

## Project Description
This project presents an innovative model for automated clinical caption generation, combining frontal chest X-ray analysis with structured patient data from radiology records.

### Key Features:
- Integration of SAT framework and GPT-3 for medical image captioning
- Generation of comprehensive radiology reports from chest X-rays
- Visualization of pathology locations using attention mechanisms

## Technical Approach

### Architecture
The model architecture consists of three main components:
1. **Encoder**: A convolutional neural network (CNN) that transforms the input image into a sequence of feature vectors
2. **Attention Module**: Focuses on specific parts of the image to identify pathologies
3. **Decoder**: Generates descriptive text based on the encoded image features

### Implementation Details
- Uses DenseNet-121, VGG-16, InceptionV3, or ResNet-101 as encoder options
- Implements adaptive average pooling for feature extraction
- Integrates GPT for enhanced natural language generation
- Provides visualization of attention maps for interpretability

## Requirements
- Python 3.8+
- PyTorch 1.9+
- Transformers library
- OpenAI API access
- NVIDIA GPU with CUDA support (recommended)

## Installation
```bash
git clone https://github.com/your-username/medical-image-captioning.git
cd medical-image-captioning
```


1. Set up your OpenAI API key:
```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
```

2. Prepare your chest X-ray images:
```python
from preprocessing import preprocess_image
processed_image = preprocess_image(image_path)
```

3. Generate captions:
```python
from model import MedicalImageCaptioner
captioner = MedicalImageCaptioner()
report = captioner.generate_report(processed_image)
```

## Important Note
**You must use your own OpenAI API key to run this code.** The repository does not include any API keys for security reasons.

## Contributions
Our contributions include:
- A novel architecture integrating SAT and GPT-3 for superior image captioning performance
- An optimized preprocessing pipeline for radiology reports
- Extensive validation across medical and general-purpose datasets




