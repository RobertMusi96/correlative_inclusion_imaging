"""CNN classification of the dataset crops (needs the optional dependencies: uv sync --extra cnn).

data   crops and masks as tensors (RGB + mask channel), normalisation, augmentation
model  ResNet-18 with an input layer for the extra mask channel
train  cross-validation over the dataset folds; run folder with curves, predictions, metrics
"""
