# Image_Segmentation
Image Segmentation on a dataset consisting of basic shapes in a contrasting background.

## dataset.py
Dataset class to generate shapes like circle, square, rectangle and triangle of varying color and thier binary mask.

Example:

![sample1](https://github.com/NeerajNixon/Image_Segmentation/assets/92161269/bfe95dba-acf9-4fe2-973a-352bba2d9af2)

![sample2](https://github.com/NeerajNixon/Image_Segmentation/assets/92161269/0ab3da73-3e99-45b7-8672-3e956876cc2c)

## model.py
The model follows the Unet architecture with 4 down sampling and up sampling layers with each layer consisting of 2 Convolution layers and ReLU function.

## metric.py
The performance of the model is measured using IoU (Intersection over Union)

## segmentation_train.py
