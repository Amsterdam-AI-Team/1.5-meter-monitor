
# Social Distancing YOLOv5

The 1.5 meter monitor is a system created to make users aware of social distancing measures.
It detects pedestrians with the open-source YOLOv5 convolutional neural network. 
It uses cheap and ubiquitous hardware, such as a simple webcam, computer with GPU (or edge devices) and any screen
to create awareness for anybody within the field of view of the camera to keep social distance. 
It does so by displaying visual information on an augmented reality interface. 
The persons that are detected are made aware of if they are keeping sufficient distance from others within their proximity.
Distances between detected persons are calculated per frame by using a calibration for the position of the camera,
an area of interest is defined and an example of 1.5x1.5 meter is given to the system as input.
By default, the calculated distances are classified as 'Safe', 'Low risk' or 'High risk' for social distancing.

The system does not store any visual information, and when detected people will be displayed with a smiley overlay to prevent visual recognition.


![](media/examples/emojis.png)

---


## Project Folder Structure

There are the following folders in the structure:
1) [`src`](./src): Folder for all source files specific to this project
3) [`weights`](./weights): Folder containing pre-trained model weights
4) [`media`](./media): Folder containing media files (icons, video)

---


## Installation

1) Clone this repository:
    ```bash
    git clone --recurse-submodules https://github.com/Amsterdam-AI-Team/1.5-meter-monitor
    ```
    If you clone the repo without the --recurse-submodules argument,
    you could also use the following command to initialize the YOLOv5 submodule:
    ```
    git submodule update --init --recursive
    ```

2) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
---


## Usage

To run with webcam:

```
$ python main.py --source 0 
```

To run with example file:

```
$ python main.py --source media/videos/example.mp4
```

---


## How it works

### Detection

For the detection of persons [YOLOv5](https://github.com/ultralytics/yolov5)  is used. YOLO (You Only Look Once), is a network for object detection.  It has fast inference speed, allowing for realtime processing and the performance is sufficient for the 1.5 meter monitor. The object detection task consists in determining the location on the image where certain objects are present, as well as classifying those objects. For the creation of the object detection model that is being used, the open dataset COCO (“Common Objects in Context”) has been used. COCO is a large-scale object detection, segmentation, and captioning dataset. It has over 330.000 images, with 250.000 people and 1.5 million objects annotated.


### Camera Calibration


![](media/examples/ROI_selection.png)

Distances are calculated by using a calibration for the position of the camera, 
a region of interest is defined and an example of 1.5x1.5 meter is given to the system as input. 
The region of interest is projected in a birds eye view and 
the 1.5x1.5 meter example is used to calculate the distance between the pedestrians.

More information and implementation details could be found in [this blog](https://blog.usejournal.com/social-distancing-ai-using-python-deep-learning-c26b20c9aa4c)

---
## Acknowledgements


Our code uses [YOLOv5](https://github.com/ultralytics/yolov5) [![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)

Some elements, e.g. the camera calibration strategy and distance calculation, are inspired by [deepak112 Social Distancing AI](https://github.com/deepak112/Social-Distancing-AI)


