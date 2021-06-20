# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:49:50 2020

@author: edperry2
"""

import os
import sys
import time
from openvino.inference_engine import IECore, IENetwork
import cv2


def prepImage(orig_image, net):
    
    ##! (2.1) Find n, c, h, w from net !##
    
    input_layer = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_layer].shape
    
    input_image = cv2.resize(orig_image, (w, h))
    input_image = input_image.transpose((2, 0, 1))
    input_image.reshape((n, c, h, w))
    return input_image, h, w

def draw_boxes(frame, result, width, height):
    '''
    Draw bounding boxes onto the frame.
    '''
    for box in result[0][0]: # Output shape is 1x1x100x7
        print(box)
        conf = box[2]
        print(conf)
        if conf >= 0.5:
            x_min = int(box[3] * width)
            y_min = int(box[4] * height)
            x_max = int(box[5] * width)
            y_max = int(box[6] * height)
            frame_crop = frame[y_min:y_max, x_min:x_max]            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    return frame, frame_crop, (y_min, y_max, x_min, x_max)

ie = IECore()
print(ie.available_devices)
print()
target_device ="CPU"
# target_device = "HETERO:CPU,GPU"
# target_device = "HETERO:CPU,GPU"

xml_path= r"C:\Users\eperr\Anaconda3\Scripts3\Edge\people_counter\models\tf\ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03\frozen_inference_graph.xml"
        
bin_path= r"C:\Users\eperr\Anaconda3\Scripts3\Edge\people_counter\models\tf\ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03\frozen_inference_graph.bin"
        
net = ie.read_network(model=xml_path, weights=bin_path)

input_layer = next(iter(net.inputs))
output_layer = next(iter(net.outputs))


image_path = r"C:\Users\eperr\Anaconda3\Scripts3\Edge\people_counter\images\sitting-on-car.jpg"

original_image = cv2.imread(image_path)
height= original_image[0]
width= original_image[1]

exec_net = ie.load_network(network=net, device_name=target_device)
pre_total =inf_total = post_total = 0

for i in range(1):
    pre_start = time.time()
    input_image, h, w = prepImage(original_image, net)
    pre_total += time.time() - pre_start

    inf_start = time.time()    
    inference_request = exec_net.infer({input_layer: input_image})
    print(inference_request[output_layer].shape)
    inf_total += time.time() - inf_start

    post_start = time.time()
    # draw_boxes(input_image, result, width, height)
    #result= exec_net.requests[0].outputs[output_layer]
    draw_boxes(input_image, inference_request[output_layer] , width, height)
    post_total += time.time() - post_start

print(pre_total, inf_total, post_total)                              