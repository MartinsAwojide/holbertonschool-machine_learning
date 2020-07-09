#!/usr/bin/env python3
"""Utils"""
import cv2
import glob
import numpy as np
import csv
import os


def load_images(images_path, as_array=True):
    """
    Loads images from a directory or file
    Args:
        images_path: is the path to a directory from which to load images
        as_array: is a boolean indicating whether the images should be loaded
        as one numpy.ndarray
            - If True, the images should be loaded as a numpy.ndarray of
                shape (m, h, w, c) where:
                - m is the number of images
                - h, w, and c are the height, width, and number of channels
                of all images, respectively
            - If False, the images should be loaded as a list of
                individual numpy.ndarrays

    Returns: images, filenames
    images is either a list/numpy.ndarray of all images
    filenames is a list of the filenames associated with each image in images
    """
    images = []
    filenames = []
    # load and save file_path and sort
    image_paths = glob.glob(images_path + "/*", recursive=False)
    image_paths.sort()

    for image_name in image_paths:
        # delete folder route just name.jpg and save in list
        names = image_name.split('/')[-1]
        # in windows
        # names = image_name.split('\\')[-1]
        filenames.append(names)

    for image_name in image_paths:
        # change the color in RGB format
        image = cv2.imread(image_name)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image_rgb)

    if as_array is True:
        images = np.array(images)
    return images, filenames


def load_csv(csv_path, params={}):
    """
    loads the contents of a csv file as a list of lists
    Args:
        csv_path: is the path to the csv to load
        params: are the parameters to load the csv with

    Returns: a list of lists representing the contents found in csv_path

    """
    path = []

    with open(csv_path, 'r') as fd:
        read = csv.reader(fd, params)
        for row in read:
            path.append(row)
    return path


def save_images(path, images, filenames):
    """
    Saves images to a specific path
    Args:
        path: is the path to the directory in which the images should be saved
        images: is a list/numpy.ndarray of images to save
        filenames: is a list of filenames of the images to save

    Returns: True on success and False on failure
    """
    if not os.path.exists(path):
        return False

    for i in range(len(images)):
        color = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(path, filenames[i]), color)
    return True


def generate_triplets(images, filenames, triplet_names):
    """
    Generates triplets
    Args:
        images: is a numpy.ndarray of shape (n, h, w, 3) containing
        the various images in the dataset
        filenames: is a list of length n containing the corresponding
        filenames for images
        triplet_names: is a list of lists where each sublist contains
        the filenames of an anchor, positive, and negative image, respectively
    Returns: a list [A, P, N]
    A is a numpy.ndarray of shape (m, h, w, 3) containing the anchor images
    P is a numpy.ndarray of shape (m, h, w, 3) containing the positive images
    N is a numpy.ndarray of shape (m, h, w, 3) containing the negative images
    """
    _, w, h, c = images.shape
    A = []
    P = []
    N = []
    for i in range(len(triplet_names)):
        anchor, pos, neg = triplet_names[i]
        anchor = anchor + ".jpg"
        pos = pos + ".jpg"
        neg = neg + ".jpg"

        if anchor in filenames:
            if pos in filenames:
                if neg in filenames:
                    idx_a = filenames.index(anchor)
                    idx_p = filenames.index(pos)
                    idx_n = filenames.index(neg)
                    img_a = images[idx_a]
                    img_p = images[idx_p]
                    img_n = images[idx_n]
                    A.append(img_a)
                    P.append(img_p)
                    N.append(img_n)
    A = [ele.reshape(1, w, h, c) for ele in A]
    A = np.concatenate(A)
    P = [ele.reshape(1, w, h, c) for ele in P]
    P = np.concatenate(P)
    N = [ele.reshape(1, w, h, c) for ele in N]
    N = np.concatenate(N)

    return [A, P, N]
