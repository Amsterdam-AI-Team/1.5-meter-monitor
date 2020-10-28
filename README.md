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


![](https://github.com/Amsterdam-AI-Team/1.5-meter-monitor/blob/master/media/examples/emojis.png)

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
    git clone https://github.com/Amsterdam-AI-Team/1.5-meter-monitor
    ```
2) Install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3) Initialize the YOLOv5 submodule
	```bash
	git submodule update --init
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

**\#TODO**


### Camera Calibration


![](https://github.com/Amsterdam-AI-Team/1.5-meter-monitor/blob/master/media/examples/ROI_selection.png)

Distances are calculated by using a calibration for the position of the camera, 
a region of interest is defined and an example of 1.5x1.5 meter is given to the system as input. 
The region of interest is projected in a birds eye view and 
the 1.5x1.5 meter example is used to calculate the distance between the pedestrians.

More information and implementation details could be found in [this blog](https://blog.usejournal.com/social-distancing-ai-using-python-deep-learning-c26b20c9aa4c)

---
## Acknowledgements


Our code uses [YOLOv5](https://github.com/ultralytics/yolov5) [![DOI](https://zenodo.org/badge/264818686.svg)](https://zenodo.org/badge/latestdoi/264818686)

Some elements, e.g. the camera calibration strategy and distance calculation, are inspired by [deepak112 Social Distancing AI](https://github.com/deepak112/Social-Distancing-AI)


