import numpy as np
import cv2
from matplotlib import pyplot as plt


def squeeze(array):
    return (array - array.min()) / (array.max() - array.min())


def squeeze_components(components, compressed):
    mins = compressed.min(axis=0)
    maxs = compressed.max(axis=0)

    components = (components - mins) / (maxs - mins)
    return np.nan_to_num(components)


def unsqueeze_components(components, compressed):
    mins = compressed.min(axis=0)
    maxs = compressed.max(axis=0)

    components = components * (maxs - mins) + mins
    return np.nan_to_num(components)


def apply_cmap(image, cmap):
    # From https://stackoverflow.com/questions/52498777/apply-matplotlib-or-custom-colormap-to-opencv-image

    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Linear range
    color_range = sm.to_rgba(np.linspace(0, 1, 256))[:, 0:3]   # drop alpha
    color_range = (color_range * 255.0).astype(np.uint8)       # [0, 1] => [0, 255]

    # Apply colormap for each channel individually
    channels = [cv2.LUT(image, color_range[:, i]) for i in range(3)]
    return np.dstack(channels)


def get_size(in_size, kernel, padding, stride):
    return ((in_size - kernel + 2 * padding) / stride) + 1
