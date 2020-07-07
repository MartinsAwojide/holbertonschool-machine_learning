#!/usr/bin/env python3
"""utils"""
import cv2
import glob
import numpy as np
import csv


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
        # for windows
        # image = cv2.imdecode(np.fromfile(image_name, np.uint8),
        #                      cv2.IMREAD_UNCHANGED)
        image = cv2.imread(image_name)

    for image_name in image_paths:
        # delete folder route just name.jpg and save in list
        names = image_name.split('/')[-1]
        # in windows
        # names = image_name.split('\\')[-1]
        filenames.append(names)

    for image_name in image_paths:
        # change the color in RGB format
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
