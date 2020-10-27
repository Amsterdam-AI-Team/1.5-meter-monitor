"""
Object detection and distance calculations for 1.5 meter monitor
"""
# pylint: disable=wrong-import-position

import sys
import os

import csv
import logging
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

import cv2
import torch
import torch.backends.cudnn as cudnn

sys.path.append("yolov5")
from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import (
    check_img_size, non_max_suppression, scale_coords, xyxy2xywh, set_logging)
from yolov5.utils.torch_utils import select_device, time_synchronized

from src.constants import CLOSENESS_LEVELS, BANNER
from src import utils
from src import plot

logger = logging.getLogger(__name__)

class SocialDistanceMonitor: # pylint: disable=too-many-instance-attributes,no-self-use,too-many-nested-blocks
    """ Code for estimating social distances from RGB cameras """
    def __init__(self, transform, opt, save_img=False):
        self.opt = opt
        self.debug = opt.debug
        self.save_txt = opt.save_txt
        self.view_img = opt.view_img
        self.transform = transform
        self.save_img = save_img
        self.fourcc = 'mp4v'
        self.resolution = (1920, 1080) # Can also be set to False to get standard camera resolution
        self.vid_writer = None
        self.vid_path = None

        self.out = self.create_dir(opt.output)
        self.webcam = self.check_webcam(opt.source)
        self.device = select_device(opt.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model, self.imgsz, self.class_names = self.load_model()
        self.overlay_images = self.get_overlay_icons()
        self.banner_icon = self.get_banner_icon()
        self.dataset = self.set_dataloader()
        self.meta_data_writer = self.get_meta_data_writer()

        set_logging()

    def detect(self): # pylint: disable=too-many-locals,too-many-branches,too-many-statements
        """ Start main code for object detection and distance calculations """
        start_time = time.time()

        logger.info('Start detecting')
        logger.debug('Device: %s', self.device)

        window_name = 'Stream'
        if self.webcam:
            # Full screen
            cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Run inference
        img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.half else None  # run once

        frame_id = 0
        for path, img, im0s, vid_cap in self.dataset:
            img, pred, prediction_time = self.get_predictions(img)

            objects_base = []
            # Process detections
            for idx_image, det in enumerate(pred):  # detections per image
                if self.webcam:  # batch_size >= 1
                    path_frame, im0 = path[idx_image], im0s[idx_image].copy()
                    print_details = '%g: ' % idx_image
                else:
                    path_frame, im0 = path, im0s
                    print_details = ''

                # Must be inside the for loop so code can be used with multiple files (e.g. images)
                save_path = str(Path(self.out) / Path(path_frame).name)

                if self.save_txt or self.debug:
                	# normalization gain whwh
                    gn_whwh = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # pylint: disable=not-callable
                    print_details += '%gx%g ' % img.shape[2:]

                if det is not None and len(det) > 0:
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    if self.save_txt or self.debug:
                        classes_cnt = Counter(det[:, -1].tolist())
                        for class_idx, class_cnt in classes_cnt.items():
                            print_details += '%g %ss, ' % (class_cnt, self.class_names[int(class_idx)])

                    # Write results
                    for *xyxy, conf, cls in det:
                        if self.save_txt:  # Write to file
                            # normalized xywh
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn_whwh).view(-1).tolist() # pylint: disable=not-callable
                            with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                                file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format

                        if self.save_img or self.view_img:  # Add bbox to image
                            label = '%s %.2f' % (self.class_names[int(cls)], conf)
                            if label is not None:
                                if (label.split())[0] == 'person':
                                    # Save bbox and initialize it with zero, the "safe" label
                                    objects_base.append([
                                        xyxy,
                                        0
                                        ])

                # Plot lines connecting people and get highest label per person
                objects_base = self.monitor_distance_current_boxes(objects_base, im0, 1)

                # Plot box with highest label on person
                plot.draw_boxes(objects_base, im0, self.overlay_images, 1,True)

                # Count label occurrences per frame
                risk_count = self.label_occurences(objects_base)

                if self.view_img:
                    # Flip screen in horizontal direction
                    im0 = im0[:, ::-1, :]

                # Plot legend
                if self.opt.add_legend:
                    im0 = plot.add_risk_counts(im0, risk_count, self.opt.lang)

                # Plot banner
                if self.opt.add_banner:
                    im0 = plot.add_banner(im0, self.banner_icon, self.opt.lang)

                if self.debug:
                    # Print frames per second
                    running_time = time.time() - start_time
                    frame_id = frame_id + 1
                    logger.debug('Frame rate: %s', round(frame_id / running_time, 2))
                    # Print time (inference + NMS)
                    logger.debug('%sDone. (%.3fs)', print_details, prediction_time)

                # Stream results
                if self.view_img:
                    if self.resolution:
                        # Interpolation INTER_AREA is better, INTER_LINEAR (default) is faster
                        im0 = cv2.resize(im0, self.resolution)
                    cv2.imshow(window_name, im0) # im0[:, ::-1, :]
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                # Save results
                if self.save_img:
                    self.save_results(im0, vid_cap, save_path)

        logger.info('Results saved to %s', Path(self.out))

        logger.info('Done. (%.3fs)', (time.time() - start_time))

    def monitor_distance_current_boxes(self, objects_base, im0_arg, thickness, dotted_line=False):
        """
        Calculate distances between the detected boxes and draw lines
        between boxes according to risk factor between two persons.
        """

        # Iterate to set label in objects_base, more advanced loop to avoid duplicates
        for i in range(len(objects_base)):
            for j in range(i,len(objects_base)):
                xyxy_1 = objects_base[i][0]
                xyxy_2 = objects_base[j][0]
                if xyxy_1 != xyxy_2:
                    # Here we will be using bottom center point of bounding box for all boxes
                    # and will transform all those bottom center points to bird eye view
                    center_low_1 = ((int(xyxy_1[2] + xyxy_1[0])) // 2, int(xyxy_1[3]))
                    center_low_2 = ((int(xyxy_2[2] + xyxy_2[0])) // 2, int(xyxy_2[3]))

                    person_points = utils.get_transformed_points((xyxy_1, xyxy_2),
                        self.transform['perspective_transform'])
                    dist = utils.calculate_real_distance(person_points[0], person_points[1],
                        self.transform['distance_w'], self.transform['distance_h'])

                    for level in sorted(CLOSENESS_LEVELS, reverse=True):
                        if dist <= CLOSENESS_LEVELS[level]['dist']:
                            # Plot distance line between two persons
                            if dotted_line:
                                # Plot dotted line
                                plot.dotline(im0_arg, center_low_1, center_low_2,
                                    CLOSENESS_LEVELS[level]['color'], thickness, 5)
                            else:
                                # Plot normal line
                                cv2.line(im0_arg, center_low_1, center_low_2,
                                    CLOSENESS_LEVELS[level]['color'], thickness)

                            # Save the label
                            objects_base[i][1] = max(objects_base[i][1], level)
                            objects_base[j][1] = max(objects_base[j][1], level)

                            break

        return objects_base

    def label_occurences(self, objects_base):
        """ Counting label occurrences """
        risk_count = Counter(sublist[1] for sublist in objects_base)

        risk_dict = {}
        for level, level_info in CLOSENESS_LEVELS.items():
            risk_dict[level_info['name']] = risk_count[level]
        risk_dict['timestamp'] = str(datetime.now())

        self.meta_data_writer.writerow(risk_dict)

        return risk_count

    def get_meta_data_writer(self):
        """ Get meta data writer """
        meta_data_path = os.path.join(self.out, datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv')
        field_names = [level['name'] for level in CLOSENESS_LEVELS.values()] + ['timestamp']
        meta_data_writer = csv.DictWriter(open(meta_data_path, 'w'), field_names)
        meta_data_writer.writeheader()
        return meta_data_writer

    def get_predictions(self, img):
        """ Get predictions """
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # unit8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            self.opt.conf_thres, self.opt.iou_thres,
            classes=self.opt.classes,
            agnostic=self.opt.agnostic_nms
        )

        t2 = time_synchronized()
        prediction_time = t2 - t1

        return img, pred, prediction_time

    def save_results(self, im0, vid_cap, save_path):
        """ Save results (image with detections) """
        if self.dataset.mode == 'images':
            cv2.imwrite(save_path, im0)
        else:
            if self.vid_path != save_path:  # new video
                self.vid_path = save_path
                if isinstance(self.vid_writer, cv2.VideoWriter):
                    self.vid_writer.release()  # release previous video writer

                # initialize the video writer
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.vid_writer = cv2.VideoWriter(
                    save_path,
                    cv2.VideoWriter_fourcc(*self.fourcc),
                    fps, (width, height))
            self.vid_writer.write(im0)

    def get_overlay_icons(self):
        """ Set icons once in memory """
        overlay_images = []

        for level in CLOSENESS_LEVELS:
            overlay_images.append(cv2.imread(CLOSENESS_LEVELS[level]['icon'], -1))

        return overlay_images

    def get_banner_icon(self):
        """ Set banner once in memory """

        banner_icon = cv2.imread(BANNER, -1)

        return banner_icon 

    def load_model(self):
        """ Load model """
        print(self.opt.weights)
        model = attempt_load(self.opt.weights, map_location=self.device)  # load FP32 model
        if self.half:
            model.half()  # to FP16

        imgsz = check_img_size(self.opt.img_size, s=model.stride.max())  # check img_size
        class_names = model.module.names if hasattr(model, 'module') else model.names

        return model, imgsz, class_names

    def set_dataloader(self):
        """ Set Dataloader """
        if self.webcam:
            self.view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(self.opt.source, img_size=self.imgsz)
        else:
            self.save_img = True
            dataset = LoadImages(self.opt.source, img_size=self.imgsz)
        return dataset

    def check_webcam(self, source):
        """ Check for webcam """
        return (
                source.isnumeric()
                or source.startswith('rtsp')
                or source.startswith('http')
                or source.endswith('.txt')
        )

    def create_dir(self, out_dir):
        """ Create output dir or empty output dir if existing """
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        return out_dir
