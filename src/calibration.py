
"""
Calibration code for estimating distances based on RGB cameras
"""

import os
import pickle as pkl

import numpy as np
import cv2

from src.constants import Color


class SizeCalibrator:
    """
    Before starting the stream and detection, collect points for region of interest for the
    bird view and for determining pixels per unit of length that are later used for
    transformations and for calculating actual distance between points
    """
    def __init__(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.mouse_pts_file = os.path.join(output_dir, 'mouse_points')

        self.mouse_pts = []
        self.image = None
        self.width, self.height = None, None
        self.perspective_transform = None
        self.distance_w, self.distance_h = None, None

    def get_mouse_points(self, event, x, y, flags, param): #pylint: disable=unused-argument,too-many-arguments
        """
        Store and color clicked points
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            num_mouse_pts = len(self.mouse_pts)
            if num_mouse_pts < 4:
                cv2.circle(self.image, (x, y), 5, Color.RED, 10)

                if num_mouse_pts >= 1:
                    previous_point = self.mouse_pts[num_mouse_pts - 1]
                    cv2.line(self.image, (x, y), previous_point, Color.GREY, 2)
                if num_mouse_pts == 3:
                    first_point = self.mouse_pts[0]
                    cv2.line(self.image, (x, y), first_point, Color.GREY, 2)

            else:
                cv2.circle(self.image, (x, y), 5, Color.BLUE, 10)

            self.mouse_pts.append((x, y))

    def get_points(self, source, force_calibrate=False):
        """
        Get first frame and wait until 8 points are selected
        Set height and width for transformation calculation
        """
        window_name = "Calibration"
        cv2.namedWindow(window_name)

        # If --source argument is an integer, make it an integer
        if source in [str(x) for x in range(0, 10)]:
            source = int(source)

        vid_cap = cv2.VideoCapture(source)

        # Get video height, width and fps
        self.height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        # Some cameras have to initialize
        for _ in range(0, 30):
            vid_cap.read()

        (_, self.image) = vid_cap.read()

        # If already calibrated, get the saved calibrations and return function
        if not force_calibrate:
            try:
                self.mouse_pts = pkl.load(open(self.mouse_pts_file, 'rb'))
                cv2.destroyWindow(window_name)
                return
            except:     #pylint: disable=bare-except
                pass

        cv2.setMouseCallback(window_name, self.get_mouse_points)

        # first frame will be used to draw ROI and horizontal and vertical 180 cm distance
        # (unit length in both directions)
        while True:
            cv2.imshow(window_name, self.image)
            cv2.waitKey(1)
            if len(self.mouse_pts) == 8: # Can also be 7
                cv2.destroyWindow(window_name)
                break

        pkl.dump(self.mouse_pts, open(self.mouse_pts_file, 'wb'))

    def calculate_transformation(self):
        """
        Points transformation from ***
        Using first 4 points or coordinates for perspective transformation.
        The region marked by these 4 points are considered ROI.
        This polygon shaped ROI is then warped into a rectangle which becomes the bird eye view.
        This bird eye view then has the property property that points are distributed uniformly
        horizontally and vertically (scale for horizontal and vertical direction will be different).
        So for bird eye view points are equally distributed, which was not case for normal view.
        """
        if self.perspective_transform is not None:
            return self.perspective_transform

        src = np.float32(np.array(self.mouse_pts[:4]))
        dst = np.float32([[0, self.height], [self.width, self.height], [self.width, 0], [0, 0]])
        self.perspective_transform = cv2.getPerspectiveTransform(src, dst)

        return self.perspective_transform

    def calculate_distances(self):
        """
        Since bird eye view has property that all points are equidistant in horizontal and vertical
        direction. distance_w and distance_h will give us a unit of distance (CALIBRATION_DISTANCE)
        in both directions (how many pixels will be there in CALIBRATION_DISTANCE-cm length
        in horizontal and vertical direction of birds eye view), which we can use to calculate
        distance between two humans in transformed view or bird eye view
        """
        if self.distance_w is not None and self.distance_h is not None:
            return self.distance_w, self.distance_h

        # using next 3 points for horizontal and vertical unit length(in this case 180 cm)

        pts = np.float32(np.array([self.mouse_pts[4:7]]))
        warped_pt = cv2.perspectiveTransform(pts, self.perspective_transform)[0]
        self.distance_w = np.sqrt(
            (warped_pt[0][0] - warped_pt[1][0]) ** 2
            + (warped_pt[0][1] - warped_pt[1][1]) ** 2
        )
        self.distance_h = np.sqrt(
            (warped_pt[0][0] - warped_pt[2][0]) ** 2
            + (warped_pt[0][1] - warped_pt[2][1]) ** 2
        )
        # pnts = np.array(self.mouse_pts[:4], np.int32)
        # cv2.polylines(self.image, [pnts], True, Color.GREY, thickness=2)

        return self.distance_w, self.distance_h
