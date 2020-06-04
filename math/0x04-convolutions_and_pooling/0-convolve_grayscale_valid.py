#!/usr/bin/env python3
"""convolve a image"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Function that performs a valid convolution on grayscale images
    Args:
        images: is a numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
                - m is the number of images
                - h is the height in pixels of the images
                - w is the width in pixels of the images
        kernel: is a numpy.ndarray with shape (kh, kw) containing the
                kernel for the convolution
                - kh is the height of the kernel
                - kw is the width of the kernel

    Returns:
            a numpy.ndarray containing the convolved images
    """
    m, image_h, image_w = images.shape
    kernel_h, kernel_w = kernel.shape
    conv_h = image_h - kernel_h + 1
    conv_w = image_w - kernel_w + 1
    # initialize the convolution array
    convolution = np.zeros((m, conv_h, conv_w))
    for height in range(conv_h):
        for width in range(conv_w):
            aux = images[:, height:kernel_h + height, width:kernel_w + width]
            convolution[:, height, width] = (np.sum(aux * kernel, axis=(2, 1)))
    # sum in 0, vertical. 1, horizontal, 2 inside, (2,1) sum inside then horiz
    return convolution
