# -*- coding: utf-8 -*-
"""
Created on Wed Sep 16 17:49:50 2020

@author: edperry2
"""

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
target_device ="CPU"
# target_device = "HETERO:CPU,GPU"
# target_device = "HETERO:CPU,GPU"

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

layers_map = ie.query_network(network=net, device_name=target_device)

for layer in layers_map:
    print("{}: {}".format(layer, layers_map[layer]))
print()

not_supported_layers = {layer for layer in net.layers.keys() if layer not in layers_map }
    
if len(not_supported_layers) != 0:
    print("The following layers are not supported by the specified device {}:\n {}"
          .format(target_device, ', '.join(not_supported_layers)))
#%%    
inference_request = exec_net.infer({input_layer: input_image})

output_layer = next(iter(net.outputs))

print(f'net.outputs = {net.outputs}')
print()
print(f'output_layer = {output_layer}')
print()
print(f'net.outputs[output_layer].shape = {net.outputs[output_layer].shape }')
print()

result = inference_request[output_layer]
print(type(result))
#%%
infer_request = exec_net.start_async(request_id=0, inputs={input_layer: input_image})

status_code = infer_request.wait()
print(f'status_code = {status_code}')
print()

output_layer = next(iter(net.outputs))

result = infer_request.outputs[output_layer]
print(type(result))
#%%
exec_net = ie.load_network(network=net, device_name=target_device, num_requests=2)

infer_request_0 = exec_net.start_async(request_id=0, inputs={input_layer: input_image})
infer_request_1 = exec_net.start_async(request_id=1, inputs={input_layer: input_image})

infer_request_0.wait()
result = infer_request_0.outputs[output_layer] 

exec_net.requests[1].wait()
result_1 = exec_net.requests[1].outputs[output_layer]

#%%
                              