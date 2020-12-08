import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

from typing import List, Dict

from detectron2.structures import BoxMode

from detectron2.data.transforms import *


def build_augmentation(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.

    Returns:
        list[Augmentation]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    augmentation = [ResizeShortestEdge(min_size, max_size, sample_style)]
    if is_train:
        pass
        augmentation.append(ResizeShortestEdge(min_size, max_size, sample_style))
        augmentation.append(RandomBrightness(0.9,1.1))
        augmentation.append(RandomContrast(0.9,1.1))
        augmentation.append(RandomRotation([-1, 1], expand=True, center=None, sample_style="range"))
        #augmentation.append(RandomFlip(prob=0.35))
    return augmentation


build_transform_gen = build_augmentation
"""
Alias for backward-compatibility.
"""