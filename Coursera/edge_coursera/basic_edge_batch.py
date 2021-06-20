# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 18:54:32 2020

@author: edperry2
"""

def  preprocessImage(images):
    pass

def preocessResult(result):
    pass

images = None

# initialize net, exec_net, input_layer, output_layer, and images array

preprocessImage(images[0])
infer_request = exec_net.start_async(request_id=0, inputs={input_layer: input_image})

# iterate over the rest of the image

for i in range(1, len(images)):
    # preprocess the current image and wait for the previous image
    input_image = preprocessImage(images[i])
    infer_request.wait()
    # Copy the result so that it is not overwritten, then start the next inference
    res = infer_request.outputs[output_layer].copy()
    infer_request = exec_net.start_async(request_id=0, inputs={input_layer: input_image})
    # postprocessing for the previous image
    preocessResult(res)
    
# process the last remaining result
infer_request.wait()
preocessResult(infer_request.outputs[output_layer])    
    