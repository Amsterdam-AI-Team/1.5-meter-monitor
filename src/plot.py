"""
Plotting module for 1.5 meter monitor
"""

import math
import cv2
import numpy as np

from src.constants import CLOSENESS_LEVELS, Color, BANNER, BANNER_WIDTH

def image_in_box(frame, box, overlay_image):
    """ Overlay a box with an image """
    x, y, width, height = box[:]
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_RGB2RGBA).copy() # Is this necesarry?
    img_small = cv2.resize(overlay_image, (width, height // 2), interpolation=cv2.INTER_AREA)
    overlay_image_alpha(
        frame,
        img_small[:, :, 0:3],
        (x, y),
        img_small[:, :, 3] / 255.0)

def overlay_image_alpha(img, img_overlay, pos, alpha_mask): #pylint: disable=too-many-locals
    """
    Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha_mask[y1o:y2o, x1o:x2o]

    for channel in range(img.shape[2]):
        img[y1:y2, x1:x2, channel] = (
                alpha * img_overlay[y1o:y2o, x1o:x2o, channel]
                + alpha_inv * img[y1:y2, x1:x2, channel])

def draw_boxes(objects_base, im0_arg, overlay_images, thickness, reference_dot=False):
    """ Draw bboxes and icons """
    for i in range(len(objects_base)):
        # Get the current bounding box
        xyxy = objects_base[i][0]
        # Convert float values to ints
        xyxy_int = [int(x) for x in xyxy[:4]]
        start_point, end_point = (xyxy_int[0], xyxy_int[1]), (xyxy_int[2], xyxy_int[3])
        xywh = (xyxy_int[0], xyxy_int[1], xyxy_int[2] - xyxy_int[0], xyxy_int[3] - xyxy_int[1])

        # Bottom center point of bounding box for reference circle
        if reference_dot:
            radius = 3
            center_low = ((int(xyxy[2] + xyxy[0])) // 2, int(xyxy[3]))

        level = objects_base[i][1]

        if level != 0:
            # Draw reference circle of starting distance line
            if reference_dot:
                cv2.circle(im0_arg, center_low, radius, CLOSENESS_LEVELS[level]['color'], thickness)

            # Draw bbox
            cv2.rectangle(im0_arg, start_point, end_point, CLOSENESS_LEVELS[level]['color'],
                thickness, cv2.LINE_AA)

        # Overlay image according to risk factor
        image_in_box(im0_arg, xywh, overlay_images[level])

def add_risk_counts(frame, risk_count, lang='EN'):
    """ Add a panel with risk counts """
    pad = np.full((140, frame.shape[1], 3), Color.WHITE, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for level in CLOSENESS_LEVELS:
        count = risk_count[level]
        color = CLOSENESS_LEVELS[level]['color']
        text = CLOSENESS_LEVELS[level]['text'][lang].upper()
        offset = 20 * level
        cv2.putText(pad, text + ": " + str(count) + " ", (50, 60+offset), font, 0.6, color, 1)

    frame = np.vstack((frame, pad))

    return frame

def add_banner(frame, banner_icon, lang='NL'):
    """ Add a banner to the screen, and rescale with padding """
    old_size = banner_icon.shape[:2] # height, width

    frame_size = frame.shape[:2]

    ratio = old_size[0]/frame_size[0]

    resized_banner_icon = cv2.resize(banner_icon,(int(old_size[0]/ratio*BANNER_WIDTH),int(old_size[1]/ratio)))

    top = int((frame_size[0]-resized_banner_icon.shape[0])/2)
    bottom = int((frame_size[0]-resized_banner_icon.shape[0])/2)
    left = 0
    right = 0

    if top + bottom + resized_banner_icon.shape[0] < frame_size[0]:
        top+=1
    if top + bottom + resized_banner_icon.shape[0] > frame_size[0]:
        top-=1

    resized_banner_icon_width_padding = cv2.copyMakeBorder(resized_banner_icon, top, bottom, left, right, cv2.BORDER_CONSTANT,value=Color.WHITE)

    frame = np.hstack((frame, resized_banner_icon_width_padding))
    return frame



def dotline(src, point_1, point_2, color, thickness, discrete_line):
    """ Draw in src image a doted line from point_1 to point_2 """

    segments = discrete_contour((point_1, point_2), discrete_line)  # Discrete the line
    for segment in segments:  # Draw the discrete line with circles
        cv2.circle(src, segment, thickness, color, -1)

    # Return tuple input point undistorted
    return src

def discrete_contour(contour, discrete_line):
    """ Takes contour points to get a number of intermediate points """

    # If contour has less of two points is not valid for operations
    if len(contour) < 2:
        print("Error: no valid segment")
        return contour

    # New contour variable
    new_contour = []

    # Iterate through all contour points
    for i, coordinate in enumerate(contour):

        # Select next contour for operation
        if not i == len(contour) - 1:
            next_coordinate = contour[i + 1]
        else:
            next_coordinate = contour[0]

        # Calculate length of segment
        segment_lenth = math.sqrt(
            (next_coordinate[0] - coordinate[0]) ** 2
            + (next_coordinate[1] - coordinate[1]) ** 2
        )

        # discrete_line: int distance to get a point by segment
        divitions = segment_lenth / discrete_line  # Number of new point for current segment
        d_y = next_coordinate[1] - coordinate[1]  # Segment's height
        d_x = next_coordinate[0] - coordinate[0]  # Segment's width

        if not divitions:
            ddy = 0  # d_y value to sum in Y axis
            ddx = 0  # d_x value to sum in X axis
        else:
            ddy = d_y / divitions  # d_y value to sum in Y axis
            ddx = d_x / divitions  # d_X value to sum in X axis

        # get new intermediate points in segment
        for j in range(0, int(divitions)):
            new_contour.append(
                (int(coordinate[0] + (ddx * j)), int(coordinate[1] + (ddy * j)))
            )

    # Return new contour with intermediate points
    return new_contour
