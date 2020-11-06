"""
Main file for 1.5 meter monitor
"""
# pylint: disable=wrong-import-position

import sys
import argparse
import logging
import time
import threading

import torch

sys.path.append("yolov5")
from yolov5.utils.general import strip_optimizer

from src.calibration import SizeCalibrator
from src.social_distance_monitor import SocialDistanceMonitor

logger = logging.getLogger(__name__)

def arg_parser():
    """ Parse command line arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/yolov5s.pt',
                        help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='media/images',
                        help='source file/folder, 0 for webcam')
    parser.add_argument('--output', type=str, default='output', help='output folder')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--classes', nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--calibrate', action='store_true', help='force re-calibration')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--add-legend', action='store_true', help='add legend with risk counts')
    parser.add_argument('--add-banner', action='store_true', help='add banner')
    parser.add_argument('--lang', type=str, default='NL', help='language for text in legend')

    return parser.parse_args()


def rotate_meta_data_log(monitor):
    """ Monitor the existing meta data log and open a new file if it becomes too large """
    allowed_size = 10**7

    while True:
        time.sleep(900)
        file_ = monitor.get_meta_data_file()
        size = file_.tell()
        if size > allowed_size:
            logger.info('Rotating existing meta data log %s; Current size is: %s', file_.name, size)
            monitor.set_meta_data_writer()


def monitor_distance(opt):
    """ Code for calling calibration and main detection function """
    logger.debug(opt)
    log_level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(level=log_level)

    calibrator = SizeCalibrator(opt.output)
    calibrator.get_points(opt.source, opt.calibrate)
    perspective_transform = calibrator.calculate_transformation()
    distance_w, distance_h = calibrator.calculate_distances()

    transform = {
        'perspective_transform': perspective_transform,
        'distance_w': distance_w,
        'distance_h': distance_h
    }

    monitor = SocialDistanceMonitor(transform, opt)

    log_monitor_thread = threading.Thread(target=rotate_meta_data_log, args=[monitor], name='MyMonitorThread')
    log_monitor_thread.start()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                monitor.detect()
                strip_optimizer(opt.weights)
        else:
            monitor.detect()


if __name__ == '__main__':
    args = arg_parser()
    monitor_distance(args)
