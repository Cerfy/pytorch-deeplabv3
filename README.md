# PyTorch-DeepLabV3

This is an implementation of Semantic Segmentation on the [Berkeley Deep Drive Dataset](http://bdd-data.berkeley.edu/index.html) for semantic segmentation. ResNet-101 was used as the feature extractor and the Atrous Spatial Pyramid Pooling module mentioned in the DeepLabV3 paper are in this repo. [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587).

The pretrained ResNet-101 model weights were taken from the MIT ImageNet. Download it [here]((http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth)) and use torch.load to load the weights into the model.

## Acknowledgement
Part of the code is borrowed from [SpeedingHZL's PyTorch-Segmentation-Toolbox](https://github.com/speedinghzl/pytorch-segmentation-toolbox). 

I have no intention of copying the models or using the code for any commercial purposes. 

I am experimenting the model on my capstone project which involves Scene Understanding of Self Driving Cars.