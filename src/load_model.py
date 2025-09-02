import torch
import tensorflow as tf

from preprocess import *


model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large', pretrained=True)
model.eval()

filename = "C:\\Users\\Vivupadi\\Downloads\\PXL_20250802_072807827.mp4"


#output_predictions = preprocess_image(filename, model)
output_predictions = preprocess_video(filename, model)


mask_image(filename, output_predictions)