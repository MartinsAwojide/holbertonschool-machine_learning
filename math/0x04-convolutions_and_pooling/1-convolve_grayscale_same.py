#!/usr/bin/env python3
"""convolve a image with padding"""
import numpy as np


def convolve_grayscale_same(images, kernel):
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

    # if odd
    padd_h = int((kernel_h - 1) / 2)
    padd_w = int((kernel_w - 1) / 2)

    # if even
    if kernel_h % 2 == 0:
        padd_h = int(kernel_h / 2)
    if kernel_w % 2 == 0:
        padd_w = int(kernel_w / 2)
    # padd only on (height, width) dimensions
    img = np.pad(images, ((0, 0), (padd_h, padd_h), (padd_w, padd_w)),
                 'constant', constant_values=0)

    convolution = np.zeros((m, image_h, image_w))

    for height in range(image_h):
        for width in range(image_w):
            aux = img[:, height:kernel_h + height, width:kernel_w + width]
            convolution[:, height, width] = (np.sum(aux * kernel, axis=(1, 2)))
    return convolution
