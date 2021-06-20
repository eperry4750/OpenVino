# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:55:21 2020

@author: edperry2
"""
#%%
import os
import sys
from openvino.inference_engine import IECore, IENetwork
import cv2


def prepImage(orig_image, net):
    
    ##! (2.1) Find n, c, h, w from net !##
    
    input_layer = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_layer].shape
    
    input_image = cv2.resize(orig_image, (w, h))
    input_image = input_image.transpose((2, 0, 1))
    input_image.reshape((n, c, h, w))
    return input_image


ie = IECore()
print(ie.available_devices)
print()
# target_device ="CPU"
# target_device = "HETERO:CPU,GPU"
# target_device = "HETERO:GPU,CPU"
target_device = "MULTI:CPU,GPU"
# target_device = "MULTI:GPU,CPU"

xml_path= r"C:\Users\eperr\Anaconda3\Scripts3\Edge\people_counter\models\tf\ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03\frozen_inference_graph.xml"
        
bin_path= r"C:\Users\eperr\Anaconda3\Scripts3\Edge\people_counter\models\tf\ssd_mobilenet_v1_0.75_depth_300x300_coco14_sync_2018_07_03\frozen_inference_graph.bin"
        
net = ie.read_network(model=xml_path, weights=bin_path)


input_layer = next(iter(net.inputs))

print(f'net.inputs = {net.inputs}')
print()
print(f'input_layer = {input_layer}')
print()
print(f'net.inputs[input_layer].shape = {net.inputs[input_layer].shape }')
print()

image_path = r"C:\Users\eperr\Anaconda3\Scripts3\Edge\people_counter\images\sitting-on-car.jpg"

original_image = cv2.imread(image_path)

##! (2.3) Preprocess the image. !##
input_image = prepImage(original_image, net)
##! (2.4) Create ExecutableNetwork object. Use the device variable for targetted device !##
exec_net = ie.load_network(network=net, device_name=target_device)
#%%
for metric in ['NETWORK_NAME','SUPPORTED_METRICS','SUPPORTED_CONFIG_KEYS','OPTIMAL_NUMBER_OF_INFER_REQUESTS']:
    print(exec_net.get_metric(metric))
    
supported_metrics = exec_net.get_metric("SUPPORTED_METRICS")
number_requests = exec_net.get_metric("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
exec_net = ie.load_network(network=net, device_name=target_device, num_requests=number_requests)
layers_map = ie.query_network(network=net, device_name=target_device)

for layer in layers_map:
    print("{}: {}".format(layer, layers_map[layer]))
print()

not_supported_layers = {layer for layer in net.layers.keys() if layer not in layers_map }
    
if len(not_supported_layers) != 0:
    print("The following layers are not supported by the specified device {}:\n {}"
          .format(target_device, ', '.join(not_supported_layers)))