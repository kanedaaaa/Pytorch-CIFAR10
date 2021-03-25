# pytorch-cifar10
Classifying CIFAR10 dataset with popular DL computer vision models.

# Requirements
[Python 3x](https://www.python.org/)

[PyTorch 1.0+](https://pytorch.org/get-started/locally/)

CUDA and proper NVIDIA drivers (*optional, only if Nvidia GPU is available*)

# Instructions
`python main.py` 

`model` folder contains net architectures, just uncomment the preferred one in `main.py`

# Current stats
Model | GPU | Accuracy | Training Time
--- | --- | --- | --- | 
[LeNet5](https://github.com/kanedaaaa/pytorch-cifar10/blob/main/models/lenet.py) | Tesla T4 | 67.15% | ...
[ResNet](https://github.com/kanedaaaa/pytorch-cifar10/blob/main/models/resnet.py) | Tesla T4 | 76.14% | 21 min
[VGG16](https://github.com/kanedaaaa/pytorch-cifar10/blob/main/models/vgg.py) | Tesla T4 | 78.60% | 6 min
VGG19 | Tesla T4 | 78.51% | 7 min

# Extras
project is no longer maintained 
