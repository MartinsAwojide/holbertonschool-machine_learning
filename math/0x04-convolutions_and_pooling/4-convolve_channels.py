#!/usr/bin/env python3
"""convolve a image with padding and stride"""
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
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
        stride: is a tuple of (sh, sw)
                - sh is the stride for the height of the image
                - sw is the stride for the width of the image

    Returns:
            a numpy.ndarray containing the convolved images
    """
    m, image_h, image_w, _ = images.shape
    kernel_h, kernel_w, _ = kernel.shape
    stride_h, stride_w = stride

    if padding == "same":
        padd_h = int((kernel_h - 1) / 2)
        padd_w = int((kernel_w - 1) / 2)
        if kernel_h % 2 == 0:
            padd_h = int(kernel_h / 2)
        if kernel_w % 2 == 0:
            padd_w = int(kernel_w / 2)
    elif padding == "valid":
        padd_h, padd_w = (0, 0)
    elif padding is tuple and len(padding) == 2:
        padd_h, padd_w = padding
    # padd only on (height, width) dimensions
    img = np.pad(images, ((0, 0), (padd_h, padd_h), (padd_w, padd_w), (0, 0)),
                 'constant', constant_values=0)

    conv_h = int((image_h + 2 * padd_h - kernel_h) / stride_h) + 1
    conv_w = int((image_w + 2 * padd_w - kernel_w) / stride_w) + 1
    convolution = np.zeros((m, conv_h, conv_w))

    for height in range(conv_h):
        for width in range(conv_w):
            aux = img[:, height * stride_h:kernel_h + height * stride_h,
                      width * stride_w:kernel_w + width * stride_w]
            convolution[:, height, width] = (np.sum(aux * kernel,
                                                    axis=(1, 2, 3)))
    return convolution
