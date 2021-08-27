#!/usr/bin/python
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils
import requests
import time
import argparse
import sys

def rem_pause():

    url = "http://192.168.1.100/sony/ircc"

    payload = "<s:Envelope\r\n    xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\"\r\n    s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\">\r\n    <s:Body>\r\n        <u:X_SendIRCC xmlns:u=\"urn:schemas-sony-com:service:IRCC:1\">\r\n            <IRCCCode>AAAAAgAAAJcAAAAZAw==</IRCCCode>\r\n        </u:X_SendIRCC>\r\n    </s:Body>\r\n</s:Envelope>"
    headers = {
    'Accept': '*/*',
    'Content-Type': 'text/xml',
    'SOAPACTION': '"urn:schemas-sony-com:service:IRCC:1#X_SendIRCC"',
    'X-Auth-PSK': '1234'
    }

    response = requests.request("POST", url, headers=headers, data = payload)

    print(response.text.encode('utf8'))

def rem_play():

    url = "http://192.168.1.100/sony/ircc"

    payload = "<s:Envelope\r\n    xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\"\r\n    s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\">\r\n    <s:Body>\r\n        <u:X_SendIRCC xmlns:u=\"urn:schemas-sony-com:service:IRCC:1\">\r\n            <IRCCCode>AAAAAgAAAJcAAAAaAw==</IRCCCode>\r\n        </u:X_SendIRCC>\r\n    </s:Body>\r\n</s:Envelope>"
    headers = {
    'Accept': '*/*',
    'Content-Type': 'text/xml',
    'SOAPACTION': '"urn:schemas-sony-com:service:IRCC:1#X_SendIRCC"',
    'X-Auth-PSK': '1234'
    }

    response = requests.request("POST", url, headers=headers, data = payload)

    print(response.text.encode('utf8'))


# parse the command line
parser = argparse.ArgumentParser(description="Locate objects in a live camera stream using an object detection DNN.", 
						   formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.detectNet.Usage())

parser.add_argument("--network", type=str, default="ssd-mobilenet-v2", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="box,labels,conf", help="detection overlay flags (e.g. --overlay=box,labels,conf)\nvalid combinations are:  'box', 'labels', 'conf', 'none'")
parser.add_argument("--threshold", type=float, default=0.5, help="minimum detection threshold to use") 
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the object detection network
net = jetson.inference.detectNet(opt.network, sys.argv, opt.threshold)

# create the camera and display
camera = jetson.utils.gstCamera(opt.width, opt.height, opt.camera)
display = jetson.utils.glDisplay()
pausepl=0
# process frames until user exits
while display.IsOpen():
	# capture the image
	img, width, height = camera.CaptureRGBA()

	# detect objects in the image (with overlay)
	detections = net.Detect(img, width, height, opt.overlay)

	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	for detection in detections:
		print(detection)
	
	if len(detections)>0 and pausepl==1:
		rem_play()
		pausepl=0
		time.sleep(1)
	
	if(len(detections)==0 and pausepl==0):
		rem_pause()
		pausepl=1
		time.sleep(1)

	# render the image
	display.RenderOnce(img, width, height)

	# update the title bar
	display.SetTitle("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
	#display.SetTitle(list.count(detections))
	# print out performance info
	net.PrintProfilerTimes()
