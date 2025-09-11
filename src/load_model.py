import torch
#import hub
import tensorflow as tf
import tensorflow_hub as hub

import cv2
import time


from preprocess import *

#Input For video based object detection
"""comment out while working with object detection on image"""
filename = "C:\\Users\\Vivupadi\\Downloads\\PXL_20250802_072807827.mp4"
##########

#Input For image based object detection
"""comment out while working with object detection on video"""
#downloaded_image_path = "C:\\Users\\Vivupadi\\Downloads\\PXL_20250603_034803966.jpg"
#downloaded_image_path = download_and_resize_image(downloaded_image_path, 1380, 950, True)
##########



#Load model
#module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
#detector = hub.load(module_handle).signatures['serving_default']

module_handle = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
detector = hub.load(module_handle)
#########



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
    breakpoint()
    end_time = time.time()

    result = {key:value.numpy() for key,value in result.items()}

    print("Found %d objects." % len(result["detection_scores"]))
    print("Inference time: ", end_time-start_time)

    boxes = np.array(result["detection_boxes"][0])  # shape: [num_detections, 4]
    classes = np.array(result["detection_classes"][0]).astype(int)  # shape: [num_detections]
    scores = np.array(result["detection_scores"][0])  # shape: [num_detections]

    #breakpoint()
    image_with_boxes = draw_boxes(img, boxes, classes, scores, label_map = coco_labels)

    display_image(image_with_boxes)

#Function to handle video input
def preprocess_video_and_run_detector(filename):
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, fFrame = video.read()
        if not ret:
            break
        
        img_rgb = cv2.cvtColor(fFrame, cv2.COLOR_BGR2RGB)

        # Expand dims to make it (1, H, W, 3)
        input_tensor = tf.expand_dims(img_rgb, axis=0)

        # Make sure it's uint8
        converted_img = tf.cast(input_tensor, tf.uint8)

        result = detector(converted_img)

        result = {key:value.numpy() for key,value in result.items()}

        boxes = np.array(result["detection_boxes"][0])  # shape: [num_detections, 4]
        classes = np.array(result["detection_classes"][0]).astype(int)  # shape: [num_detections]
        scores = np.array(result["detection_scores"][0])  # shape: [num_detections]

        #breakpoint()
        image_with_boxes = draw_boxes(fFrame, boxes, classes, scores, label_map = coco_labels)

        #display_image(image_with_boxes)
        display_video_object(image_with_boxes)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    video.release()
    cv2.destroyAllWindows()

coco_labels= load_labels('coco_label.txt')


#To detect Images(comment out while working with object detection on video)
#run_detector(detector, downloaded_image_path)

#To detect objects in videos (comment out while working with object detection on image)
preprocess_video_and_run_detector(filename)