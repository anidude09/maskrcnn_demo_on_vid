import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

def get_engine():
    """Returns the appropriate Keras engine."""
    return keras.layers

def load_weights(model_path, by_name=True):
    """Load weights from h5 file."""
    import h5py
    from tensorflow.keras.layers import Dense, Conv2D
    
    with h5py.File(model_path, mode='r') as f:
        # List all groups
        print("Keys: %s" % f.keys())
        
def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image."""
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                 image[:, :, c] * (1 - alpha) + alpha * color[c],
                                 image[:, :, c])
    return image

class Config(object):
    """Base configuration class."""
    NAME = None
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50
    BACKBONE = "resnet50"
    NUM_CLASSES = 1 + 80  # Background + objects
    
    # Input image size
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    
    # RPN parameters
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    
    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT

class CocoConfig(Config):
    """Configuration for COCO dataset."""
    NAME = "coco"
    IMAGES_PER_GPU = 2
    NUM_CLASSES = 1 + 80  # COCO has 80 classes 