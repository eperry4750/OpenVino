# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:47:20 2020

@author: edperry2
"""

import os
import sys
from openvino.inference_engine import IECore, IENetwork
import cv2

# Prepares image for imference
# inputs:
#     orig_image - numpy array containing the original, unprocessed image
#     net        - IENetwork object
# output: 
#     preprocessed image.
def prepImage(orig_image, net):
    
    ##! (2.1) Find n, c, h, w from net !##
    
    input_layer = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_layer].shape
    
    input_image = cv2.resize(orig_image, (w, h))
    input_image = input_image.transpose((2, 0, 1))
    input_image.reshape((n, c, h, w))
    return input_image

# Processes the result. Prints the number of detected vehices.
# inputs:
#    detected_obects - numpy array containing the ooutput of the model
#    prob_threashold - Required probability for "detection"
# output:
#    numpy array of image wtth rectangles drawn around the vehicles.
def printCount(detected_objects, prob_threshold=0.5):
    detected_count = 0
    for obj in detected_objects[0][0]:
        # Draw only objects when probability more than specified threshold
        if obj[2] > prob_threshold:
            detected_count+=1    
    print("{} vehicles detected.".format(detected_count))

# Getting the device as commandline argument
device = sys.argv[1]
    
##! (2.2) create IECore and IENetwork objects for vehicle-detection-adas-0002 !##
ie = IECore()

xml_path="/data/intel/vehicle-detection-adas-0002/FP16-INT8/vehicle-detection-adas-0002.xml"
bin_path="/data/intel/vehicle-detection-adas-0002/FP16-INT8/vehicle-detection-adas-0002.bin"

net = ie.read_network(model=xml_path, weights=bin_path)
input_layer = next(iter(net.inputs))

image_path = "cars_1900_first_frame.jpg"
original_image = cv2.imread(image_path)

##! (2.3) Preprocess the image. !##
prep_image = prepImage(original_image, net)
##! (2.4) Create ExecutableNetwork object. Use the device variable for targetted device !##
exec_net = ie.load_network(network=net, device_name="CPU")
##! (2.5) Run synchronous inference. !##
inference_request =exec_net.infer({input_layer: prep_image})
##! (2.6) Run printCount. Make sure you extracted the array result form the dictionary returned by infer(). !##
output_layer = next(iter(net.outputs))
detected_objects = inference_request.outputs[output_layer]
printCount(detected_objects, prob_threshold=0.5)

