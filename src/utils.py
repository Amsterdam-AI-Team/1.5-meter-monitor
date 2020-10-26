"""
Utils module for 1.5 meter monitor
"""

import cv2
import numpy as np

from src.constants import CALIBRATION_DISTANCE

def get_transformed_points(boxes, perspective_transform):
    """
    Function to calculate bottom center for all bounding boxes (position where person stands)
    and transform perspective for all points.
    """
    bottom_points = []
    for box in boxes:
        pnts = np.array([[[int(box[0] + (box[2] * 0.5)), int(box[1] + box[3])]]], dtype="float32")
        bd_pnt = cv2.perspectiveTransform(pnts, perspective_transform)[0][0]
        pnt = (int(bd_pnt[0]), int(bd_pnt[1]))
        bottom_points.append(pnt)

    return bottom_points


def calculate_real_distance(point1, point2, distance_w, distance_h):
    """
    Function calculates distance between two points(humans).
    distance_w, distance_h represents number of pixels in one unit length (CALIBRATION_DISTANCE)
    horizontally and vertically. We calculate horizontal and vertical distance in pixels
    for two points and get ratio in terms of the unit distance using distance_w and distance_h.
    Then we calculate how much cm distance is horizontally and vertically and then
    using pythagoras we calculate distance between points in terms of cm.
    """
    height = abs(point2[1]-point1[1])
    width = abs(point2[0]-point1[0])

    dis_w = float((width/distance_w)*CALIBRATION_DISTANCE)
    dis_h = float((height/distance_h)*CALIBRATION_DISTANCE)

    return int(np.sqrt((dis_h ** 2) + (dis_w ** 2)))
