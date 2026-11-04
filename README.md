# Breast Ultrasound Image Tumor Segmentation
### A Comparative Study of Attention U-Net++ and Residual Attention U-Net++

## Overview

This project explores deep learning-based breast lesion segmentation using the Breast Ultrasound Images Dataset (BUSI). Two encoder-decoder architectures based on U-Net++ are implemented and evaluated:

- **Attention U-Net++** — A nested U-Net architecture enhanced with attention gates to focus on relevant lesion regions.
- **Residual Attention U-Net++** — An enhanced architecture incorporating residual convolutional blocks, nested skip connections, and attention mechanisms to improve feature representation and segmentation performance.

The primary objective is to investigate the effectiveness of architectural enhancements in segmenting breast lesions from ultrasound images, particularly in the presence of complex boundaries, low contrast, and background noise.

Both models are implemented using TensorFlow and Keras.


## Models

### 1. Attention U-Net++

The first model combines the nested skip connections of U-Net++ with attention gates.

**Key components:**
- Standard convolutional blocks with two 3×3 convolutions.
- Batch Normalization and ReLU activation.
- Dropout for regularization.
- Nested skip connections for multi-scale feature aggregation.
- Attention gates to suppress irrelevant background features.
- Transposed convolutions for upsampling.

### 2. Residual Attention U-Net++

The second model extends the architecture with residual convolutional blocks.

**Key components:**
- Residual convolutional blocks with identity projection shortcuts.
- Batch Normalization and ReLU activation.
- Nested skip connections.
- Attention gates at decoder stages.
- Multi-scale feature fusion.
- Dropout-based regularization.

The residual connections facilitate feature propagation through deeper network stages, while attention gates help the model focus on lesion-related spatial regions.

## Dataset

The models are trained using the **Breast Ultrasound Images Dataset (BUSI)** introduced by Al-Dhabyani et al.

The dataset contains ultrasound images categorized into three groups:

| Category | Number of Images |
|---|---:|
| Benign | 437 |
| Malignant | 210 |
| Normal | 133 |
| Total | 780 |

Segmentation masks are provided for lesion-containing images. Multiple masks corresponding to the same image are merged into a single binary mask.

Dataset source: [BUSI Dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

### Preprocessing

The following preprocessing steps are applied:

- Images are loaded in RGB format.
- Corresponding lesion masks are merged using pixel-wise maximum.
- Images and masks are resized to 256 × 256 pixels using padding to preserve aspect ratio.
- Pixel values are normalized to the range [0, 1].
- A subset of normal images is sampled to reduce class imbalance.

## Data Augmentation

The Residual Attention U-Net++ model employs online data augmentation using Albumentations.

Applied transformations include:

- Horizontal flipping.
- Vertical flipping.
- Random shifts.
- Random scaling.
- Random rotations.
- Elastic transformations.
- Grid distortion.

These transformations are applied consistently to both images and segmentation masks.


### Loss Function

Both models use a weighted hybrid loss function combining Binary Cross-Entropy and Dice Loss.

The weighted BCE assigns a higher penalty to errors on lesion pixels, while the Dice component encourages overlap between predicted and ground-truth masks.

### Evaluation Metrics

The following metrics are used during training:

- Dice Coefficient
- Pixel Accuracy
- Validation Loss

Dice Coefficient is particularly relevant for segmentation tasks involving imbalanced foreground and background regions.

## Experimental Results

The following results are extracted from the training logs.

| Metric | Attention U-Net++ | Residual Attention U-Net++ |
|---|---:|---:|
| Best reported validation Dice | 0.6037 | 0.7139 |
| Best reported validation Accuracy | 0.9587 | 0.9715 |
| Training Dice (reported final epoch) | 0.8469 | 0.8006 |
| Maximum training epochs | 60 | 60 |

The Residual Attention U-Net++ achieved a higher reported validation Dice score, suggesting improved lesion segmentation performance under its training configuration.


### Requirements

- Python 3.10+
- TensorFlow
- NumPy
- OpenCV
- Albumentations
- scikit-learn
- Matplotlib
