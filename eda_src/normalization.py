import os
import pickle as pkl

import numpy as np
import torch


__all__ = ['min_max_normalization']

def min_max_normalization(image):
    '''
    Min-Max normalization to move the pixel values to [0,1]
    '''
    image = (image - image.min()) / (image.max() - image.min())

    # If the image is a tensor image.float, else image.astype(np.float32)
    if isinstance(image, torch.Tensor):
        image = image.float()
    elif isinstance(image, np.ndarray):
        image = image.astype(np.float32)
        
    return image
