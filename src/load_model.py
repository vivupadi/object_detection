import torch
#import hub
import tensorflow_hub as hub

import cv2
import time
import tensorflow as tf

from preprocess import *


module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"

detector = hub.load(module_handle).signatures['serving_default']

# By Heiko Gorski, Source: https://commons.wikimedia.org/wiki/File:Naxos_Taverna.jpg
#image_url = "https://upload.wikimedia.org/wikipedia/commons/6/60/Naxos_Taverna.jpg" 
image_url = "https://upload.wikimedia.org/wikipedia/commons/9/99/Golden_Retriever_Carlos_%2810560990993%29.jpg"
downloaded_image_path = "C:\\Users\\Vivupadi\\Downloads\\_DSC8706.JPG"
#downloaded_image_path = download_and_resize_image(image_url, 1280, 856, True)


def load_img(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    return img

def load_labels(path = 'coco_label.txt'):
    with open(path, 'r') as f:
        label = f.read().splitlines()
    return label

def run_detector(detector, path):
    img = load_img(path)

    converted_img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
    start_time = time.time()
    
    # Load image using OpenCV
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Expand dims to make it (1, H, W, 3)
    input_tensor = tf.expand_dims(img_rgb, axis=0)

    # Make sure it's uint8
    converted_img = tf.cast(input_tensor, tf.uint8)

    result = detector(converted_img)
    #breakpoint()
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time-start_time)

    boxes = np.array(result["detection_boxes"][0])  # shape: [num_detections, 4]
    classes = np.array(result["detection_classes"][0]).astype(int)  # shape: [num_detections]
    scores = np.array(result["detection_scores"][0])  # shape: [num_detections]

    #breakpoint()
    image_with_boxes = draw_boxes(img, boxes, classes, scores, label_map = coco_labels)
    
    breakpoint()

    display_image(image_with_boxes)

coco_labels= load_labels('coco_label.txt')

run_detector(detector, downloaded_image_path)