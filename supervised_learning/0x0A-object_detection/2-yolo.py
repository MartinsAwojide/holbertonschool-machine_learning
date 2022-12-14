#!/usr/bin/env python3
"""Yolo class"""
import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class"""
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor
        Args:
            model_path: is the path to where a Darknet Keras model is stored
            classes_path: is the path to where the list of class names used
            for the Darknet model, listed in order of index, can be found
            class_t: is a float representing the box score threshold for
            the initial filtering step
            nms_t: float representing the IOU threshold for non-max suppression
            anchors: is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
                - outputs is the number of outputs (predictions) made by the
                Darknet model
                - anchor_boxes is the number of anchor boxes used for each
                prediction
                - 2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as fd:
            classes = fd.read()
            classes = classes.split('\n')
            if len(classes[-1]) == 0:
                classes = classes[:-1]

        self.class_names = classes
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """calculates a sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        process outputs
        Args:
            outputs:is a list of numpy.ndarrays containing the predictions from
            the Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
            grid_height & grid_width => the height and width of the grid used
            for the output
            anchor_boxes => the number of anchor boxes used
            4 => (t_x, t_y, t_w, t_h)
            1 => box_confidence
            classes => class probabilities for all classes
            image_size:is a numpy.ndarray containing the image???s original
            size [image_height, image_width]

        Returns: a tuple of (boxes, box_confidences, box_class_probs)
            - boxes: a list of numpy.ndarrays of shape (grid_height, grid_width
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively:
            4 => (x1, y1, x2, y2)
            (x1, y1, x2, y2) should represent the boundary box relative to
            original image
            - box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the box confidences
            for each output, respectively
            - box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the box???s class
            probabilities for each output, respectively

        """
        boxes = []
        image_h, image_w = image_size
        confidence_boxes = []
        prop_boxes = []

        for i, output in enumerate(outputs):
            # there is three outputs one for each grids 13x13, 26x26, 52x52
            grid_h, grid_w, n_anchor, _ = outputs[i].shape
            box = np.zeros((grid_h, grid_w, n_anchor, 4))
            # get coordinates, width height of the outputs
            tx = (output[:, :, :, 0])
            ty = (output[:, :, :, 1])
            tw = (output[:, :, :, 2])
            th = (output[:, :, :, 3])

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            # normalize
            tx_n = self.sigmoid(tx)
            ty_n = self.sigmoid(ty)

            # get corners of grid
            cx = np.tile(np.arange(0, grid_w), grid_h)
            cx = cx.reshape(grid_w, grid_w, 1)

            cy = np.tile(np.arange(0, grid_w), grid_h)
            # y doesn't change until x finish so .T
            cy = cy.reshape(grid_h, grid_h).T
            cy = cy.reshape(grid_h, grid_h, 1)

            # boxes prediction
            bx = tx_n + cx
            by = ty_n + cy
            bw = np.exp(tw) * pw
            bh = np.exp(th) * ph

            # normalize
            bx /= grid_w
            by /= grid_h
            bw /= self.model.input.shape[1].value
            bh /= self.model.input.shape[2].value

            # bounding box coordinates respect image. top left corner (x1, y1)
            # and bottom right corner (x2, y2)
            x1 = (bx - (bw / 2)) * image_w
            y1 = (by - (bh / 2)) * image_h
            x2 = (bx + (bw / 2)) * image_w
            y2 = (by + (bh / 2)) * image_h
            box[:, :, :, 0] = x1
            box[:, :, :, 1] = y1
            box[:, :, :, 2] = x2
            box[:, :, :, 3] = y2
            boxes.append(box)

            aux = output[:, :, :, 4]
            confidence = self.sigmoid(aux)
            confidence = confidence.reshape(grid_h, grid_w, n_anchor, 1)
            confidence_boxes.append(confidence)

            aux = output[:, :, :, 5:]
            class_props = self.sigmoid(aux)
            prop_boxes.append(class_props)
        return boxes, confidence_boxes, prop_boxes

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Select higher probability
        Args:
            boxes: boxes: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 4)
            containing processed boundary boxes for each output, respectively
            box_confidences: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, 1) containing
            the processed box confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape
            (grid_height, grid_width, anchor_boxes, classes) containing
            the processed box class probabilities for each output, respectively

        Returns: a tuple of (filtered_boxes, box_classes, box_scores)
        filtered_boxes: a numpy.ndarray of shape (?, 4)
        containing all of the filtered bounding boxes:
        - box_classes: a numpy.ndarray of shape (?,) containing the class
        number that each box in filtered_boxes predicts, respectively
        - box_scores: a numpy.ndarray of shape (?) containing the box scores
        for each box in filtered_boxes, respectively
        """

        multiply = []
        # take one of each and multiply
        for confidence, classes in zip(box_confidences, box_class_probs):
            multiply.extend(confidence * classes)

        # from list of numpy arrays to one numpy array
        multiply = np.concatenate(multiply)

        # argmax and max to take the highest probability
        index = np.argmax(multiply, -1)
        score_class = np.max(multiply, -1)

        # reshape(-1) turn them into a vector
        index = index.reshape(-1)
        score_class = score_class.reshape(-1)

        # mask is going to return a list with the positions
        # that fulfill the condition
        mask = np.where(score_class >= self.class_t)

        box_class = index[mask]
        box_score = score_class[mask]

        # processing the boxes, turn them into matrix of (?, 4)
        filter_box = [elem.reshape(-1, 4) for elem in boxes]
        filter_box = np.concatenate(filter_box)
        filter_box = filter_box[mask]

        return filter_box, box_class, box_score
