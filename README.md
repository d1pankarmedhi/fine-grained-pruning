# PyTorch Fine Grained Pruning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.9+](https://img.shields.io/badge/pytorch->=1.9-orange.svg)](https://pytorch.org/)

## Overview

This repository provides a practical implementation of fine-grained magnitude-based pruning for a ResNet-like Convolutional Neural Network (CNN) in PyTorch. The goal is to demonstrate and explore model compression techniques to significantly reduce model size and computational cost while maintaining acceptable accuracy levels.

**Key Features:**

- **ResNet-like Architecture:** Implements a ResNet-inspired CNN tailored for CIFAR-10 image classification.
- **Fine-Grained Pruning:** Utilizes magnitude-based pruning to achieve sparsity at the individual weight level, maximizing compression potential.
- **Sensitivity Scan:** Includes a sensitivity analysis tool to determine the robustness of each layer to pruning, guiding effective sparsity allocation.
- **Comprehensive Evaluation:** Provides metrics for model size reduction, sparsity, and accuracy, allowing for a thorough understanding of the pruning impact.
- **Fine-tuning for Recovery:** Implements fine-tuning techniques to recover accuracy lost due to pruning, balancing compression and performance.

### Dataset

The code automatically downloads the CIFAR-10 dataset to the `data/cifar10` directory upon the first run.

## Experiment

### Model

**ResNet** like model summary:

```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
├─Sequential: 1-1                        [-1, 512, 2, 2]           --
|    └─Conv2d: 2-1                       [-1, 64, 32, 32]          1,728
|    └─BatchNorm2d: 2-2                  [-1, 64, 32, 32]          128
|    └─ReLU: 2-3                         [-1, 64, 32, 32]          --
|    └─Conv2d: 2-4                       [-1, 128, 32, 32]         73,728
|    └─BatchNorm2d: 2-5                  [-1, 128, 32, 32]         256
|    └─ReLU: 2-6                         [-1, 128, 32, 32]         --
|    └─ResidualBlock: 2-7                [-1, 128, 32, 32]         295,424
|    └─MaxPool2d: 2-8                    [-1, 128, 16, 16]         --
|    └─Conv2d: 2-9                       [-1, 256, 16, 16]         294,912
|    └─BatchNorm2d: 2-10                 [-1, 256, 16, 16]         512
|    └─ReLU: 2-11                        [-1, 256, 16, 16]         --
|    └─Conv2d: 2-12                      [-1, 256, 16, 16]         589,824
|    └─BatchNorm2d: 2-13                 [-1, 256, 16, 16]         512
|    └─ReLU: 2-14                        [-1, 256, 16, 16]         --
|    └─ResidualBlock: 2-15               [-1, 256, 16, 16]         1,180,672
|    └─MaxPool2d: 2-16                   [-1, 256, 8, 8]           --
|    └─Conv2d: 2-17                      [-1, 512, 8, 8]           1,179,648
|    └─BatchNorm2d: 2-18                 [-1, 512, 8, 8]           1,024
|    └─ReLU: 2-19                        [-1, 512, 8, 8]           --
|    └─Conv2d: 2-20                      [-1, 512, 8, 8]           2,359,296
|    └─BatchNorm2d: 2-21                 [-1, 512, 8, 8]           1,024
|    └─ReLU: 2-22                        [-1, 512, 8, 8]           --
|    └─ResidualBlock: 2-23               [-1, 512, 8, 8]           4,720,640
|    └─MaxPool2d: 2-24                   [-1, 512, 4, 4]           --
|    └─Conv2d: 2-25                      [-1, 512, 4, 4]           2,359,296
|    └─BatchNorm2d: 2-26                 [-1, 512, 4, 4]           1,024
|    └─ReLU: 2-27                        [-1, 512, 4, 4]           --
|    └─Conv2d: 2-28                      [-1, 512, 4, 4]           2,359,296
|    └─BatchNorm2d: 2-29                 [-1, 512, 4, 4]           1,024
|    └─ReLU: 2-30                        [-1, 512, 4, 4]           --
|    └─ResidualBlock: 2-31               [-1, 512, 4, 4]           4,720,640
|    └─MaxPool2d: 2-32                   [-1, 512, 2, 2]           --
├─Linear: 1-2                            [-1, 10]                  5,130
==========================================================================================
Total params: 20,145,738
Trainable params: 20,145,738
Non-trainable params: 0
Total mult-adds (G): 1.62
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 13.50
Params size (MB): 76.85
Estimated Total Size (MB): 90.36
==========================================================================================
```

Model trained on **CIFAR-10** dataset for for 50 epochs achieving accuracy of _45.79%_. It is possible to optimize the training process and train the model for more epochs to reach a peak accuracy but that is not the goal of this project.

### Pruning

**Fine grained pruning:** It is a pruning technique that mostly focuses on selectively removing individual parameters from a model across different layers. You can also refer to this as _Unstructured Pruning_.

**Layer Sensitivity:** Layer sensitivity shows how each layer performs for different sparsity ratios. Depending upon the change in the accuracy, the sparsity ratio for the respective layer is set.

<img src = "https://github.com/user-attachments/assets/63a5043a-33bf-40ab-8f71-663bedbe371e">

```python
# sparsity ratio
{
    'backbone.conv0.weight': 0.0,
    'backbone.conv1.weight': 0.6,
    'backbone.residual_block0.conv1.weight': 0.8,
    'backbone.residual_block0.conv2.weight': 0.8,
    'backbone.conv2.weight': 0.6,
    'backbone.conv3.weight': 0.7,
    'backbone.residual_block1.conv1.weight': 0.8,
    'backbone.residual_block1.conv2.weight': 0.9,
    'backbone.conv4.weight': 0.7,
    'backbone.conv5.weight': 0.7,
    'backbone.residual_block2.conv1.weight': 0.9,
    'backbone.residual_block2.conv2.weight': 0.9,
    'backbone.conv6.weight': 0.7,
    'backbone.conv7.weight': 0.8,
    'backbone.residual_block3.conv1.weight': 0.8,
    'backbone.residual_block3.conv2.weight': 0.8,
    'classifier.weight': 0.0
}
```

Pruning results in reduced model complexity and size.

```
Original model size: 76.85 MiB
Sparse model size: 16. 16 MiB (4.76x smaller)
```

### Fine Tuning

Fine tuning model for 5 epochs has recovered the model accuracy, surpassing the original accuracy.

```
Finetuning Fine-grained Pruned Sparse Model
--------------------------------------------------
Epoch 1 Accuracy 40.05% / Best Accuracy: 40.05%
Epoch 2 Accuracy 47.08% / Best Accuracy: 47.08%
Epoch 3 Accuracy 51.54% / Best Accuracy: 51.54%
Epoch 4 Accuracy 52.70% / Best Accuracy: 52.70%
Epoch 5 Accuracy 55.97% / Best Accuracy: 55.97%
```

Original model was probably overfitting due to its complexity. Reducing the parameters and complexity has helped the model learn better and fit to the data.

### Result

```
Original model size: 76.85 MiB
Original model accuracy (50 epochs): 45.79%
Sparsed model size: 16.16 MiB (21% of original model)
Sparsed model accuracy (5 epochs): 55.97%
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project is inspired by `TinyML and Efficient Deep Learning Computing` by **Hanlab MIT**.
