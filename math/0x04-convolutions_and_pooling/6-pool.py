#!/usr/bin/env python3
"""convolve a image with pooling"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Function that performs a valid convolution on grayscale images
    Args:
        images: is a numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
                - m is the number of images
                - h is the height in pixels of the images
                - w is the width in pixels of the images
        kernel_shape:  is a tuple of (kh, kw) containing the kernel
                shape for the pooling
                - kh is the height of the kernel
                - kw is the width of the kernel
        stride: is a tuple of (sh, sw)
                - sh is the stride for the height of the image
                - sw is the stride for the width of the image
        mode: indicates the type of pooling
                - max indicates max pooling
                - avg indicates average pooling

    Returns:
            a numpy.ndarray containing the pooled images
    """
    m, image_h, image_w, nx = images.shape
    kernel_h, kernel_w, = kernel_shape
    stride_h, stride_w = stride

    pool_h = int((image_h - kernel_h) / stride_h) + 1
    pool_w = int((image_w - kernel_w) / stride_w) + 1

    pool = np.zeros((m, pool_h, pool_w, nx))

    for height in range(pool_h):
        for width in range(pool_w):
            aux = images[:, height * stride_h:kernel_h + height * stride_h,
                         width * stride_w:kernel_w + width * stride_w]

            if mode == 'max':
                pool[:, height, width] = np.max(aux, axis=(1, 2))
            if mode == 'avg':
                pool[:, height, width] = np.mean(aux, axis=(1, 2))
    return pool
