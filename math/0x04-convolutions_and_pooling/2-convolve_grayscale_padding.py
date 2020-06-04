#!/usr/bin/env python3
"""convolve a image with padding"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
        padding: is a tuple of (ph, pw)
                - ph is the padding for the height of the image
                - pw is the padding for the width of the image
                - the image should be padded with 0’s

        kernel: padding is either a tuple of (ph, pw), ‘same’, or ‘valid

    Returns:
            a numpy.ndarray containing the convolved images
    """
    m, image_h, image_w = images.shape
    kernel_h, kernel_w = kernel.shape

    padd_h, padd_w = padding
    # padd only on (height, width) dimensions
    img = np.pad(images, ((0, 0), (padd_h, padd_h), (padd_w, padd_w)),
                 'constant', constant_values=0)

    conv_h = image_h + 2 * padd_h - kernel_h + 1
    conv_w = image_w + 2 * padd_w - kernel_w + 1
    convolution = np.zeros((m, conv_h, conv_w))

    for height in range(conv_h):
        for width in range(conv_w):
            aux = img[:, height:kernel_h + height, width:kernel_w + width]
            convolution[:, height, width] = (np.sum(aux * kernel, axis=(1, 2)))
    return convolution
