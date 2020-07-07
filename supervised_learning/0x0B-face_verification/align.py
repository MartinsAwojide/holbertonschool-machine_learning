#!/usr/bin/env python3
"""Class FaceAlign"""
import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt


class FaceAlign:
    """Class FaceAlign"""
    def __init__(self, shape_predictor_path):
        """
        Class constructor
        Args:
            shape_predictor_path: shape_predictor_path is the path to
            the dlib shape predictor model
        """
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(shape_predictor_path)

    def detect(self, image):
        """
        Detects a face in an image
        Args:
            image: s a numpy.ndarray of rank 3 containing an image
            from which to detect a face

        Returns:  a dlib.rectangle containing the boundary box for
        the face in the image, or None on failure

        """
        try:
            # detect faces and make a rectangle around
            faces = self.detector(image, 1)
            area_aux = 0
            # if no faces return rectangel as image
            rect = dlib.rectangle(0, 0, image.shape[1], image.shape[0])
            for face in faces:
                # if many rectangles(faces) return the largest only
                if face.area() > area_aux:
                    area_aux = face.area()
                    rect = face
            return rect
        except RuntimeError:
            return None

    def find_landmarks(self, image, detection):
        """
        Finds facial landmarks
        Args:
            image: numpy.ndarray of image from which to find facial landmarks
            detection: is a dlib.rectangle containing the boundary box of
            the face in the image

        Returns: a numpy.ndarray of shape (p, 2)containing the landmark points,
        or None on failure
                - p is the number of landmark points
                - 2 is the x and y coordinates of the point
        """
        try:
            shape = self.shape_predictor(image, detection)
            print(shape)
            coords = np.zeros((shape.num_parts, 2), dtype="int")
            # dlib.num_parts
            for i in range(0, shape.num_parts):
                x = shape.part(i).x
                y = shape.part(i).y
                coords[i] = [x, y]
            return coords
        except RuntimeError:
            return None

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        Aligns an image for face verification
        Args:
            image: numpy.ndarray of rank 3 containing the image to be aligned
            landmark_indices: is a numpy.ndarray of shape (3,) containing the
            indices of the three landmark points that should be used for the
            affine transformation
            anchor_points: is a numpy.ndarray of shape (3, 2) containing the
            destination points for the affine transformation,
            scaled to the range [0, 1]
            size: is the desired size of the aligned image

        Returns:
        a numpy.ndarray of shape (size, size, 3) containing the aligned image,
        or None if no face is detected
        """
        try:
            rectangle = self.detect(image)
            landmarks = self.find_landmarks(image, rectangle)
            affine = landmarks[landmark_indices]
            affine = affine.astype('float32')
            anchors = anchor_points * size
            warp_mat = cv2.getAffineTransform(affine, anchors)
            warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

            return warp_dst

        except RuntimeError:
            return None
