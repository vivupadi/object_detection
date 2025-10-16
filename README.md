![License](https://img.shields.io/badge/License-MIT-green.svg)

# Realtime Object detection on Videos
A realtime frame-by-frame object detection on user-given video input.

## Detection Classes
The model currently detects following 80 objects in the image/video.
Labels of the objects can be seen here: [Labels](src/coco_label.txt)

## 🛠️Tech Stack  
-OpenCV

-PyTorch

-Tensorflow Lite

-Python

-**Version Control**: Git & GitHub

## Model Used

SSD Mobilenet V2 Object detection model, trained on COCO 2017 dataset. A small and fast model: ssd_mobilenet_v2

- Inputs
A three-channel image of variable size - the model does NOT support batching. The input tensor is a tf.uint8 tensor with shape [1, height, width, 3] with values in [0, 255].

- Output
The output dictionary contains:

  -- num_detections: a tf.int tensor with only one value, the number of detections [N].
  
  -- detection_boxes: a tf.float32 tensor of shape [N, 4] containing bounding box coordinates in the following order: [ymin, xmin, ymax, xmax].
  
  -- detection_classes: a tf.int tensor of shape [N] containing detection class index from the label file.
  
  -- detection_scores: a tf.float32 tensor of shape [N] containing detection scores.
  
  -- raw_detection_boxes: a tf.float32 tensor of shape [1, M, 4] containing decoded detection boxes without Non-Max suppression. M is the number of raw detections.
  
  -- raw_detection_scores: a tf.float32 tensor of shape [1, M, 90] and contains class score logits for raw detection boxes. M is the number of raw detections.
  
  -- detection_anchor_indices: a tf.float32 tensor of shape [N] and contains the anchor indices of the detections after NMS.
  
  -- detection_multiclass_scores: a tf.float32 tensor of shape [1, N, 91] and contains class score distribution (including background) for detection boxes in the image including background class.

Source:
https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2

## Installation on Local System
### Clone the repository

git clone https://github.com/vivupadi/object_detection.git

cd object_detection/src

### Create virtual environment

python -m venv venv

source venv/bin/activate # On Windows: venv\Scripts\activate

### Install dependencies

pip install -r requirements.txt

### Run the Application

python load_model.py

## Model architecture

Minimum Score Threshold: 55%

## Inference on CPU

### Object detection on Video
![til](https://github.com/vivupadi/object_detection/blob/main/Data/Obj_detect.gif)


### Object detection on Image
![til](https://github.com/vivupadi/object_detection/blob/main/Data/obj_detect.jpg)
** Model is not fine-tunable! 

# Future Plans

## Quantization


## Inference on Raspberry Pi 4 + Camera module


## Structure of Pipeline


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

<div align="center">
⭐ Star this repo if you find it helpful!
  
Made with ❤️ by Vivek Padayattil
</div>
