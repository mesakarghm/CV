from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
import argparse
import os 
import os.path as osp
from darknet import Darknet
import pickle
import pandas as pd
import random

def arg_parse():
	"""
	Parse the arguments given to the detection module
	"""

	parser = argparse.ArgumentParser(description = "Real Time Video Detection module for YOLO v3")
	parser.add_argument("--bs", dest = "bs", help = "Batch Size", default = 1)
	parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
	parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS threshold", default = 0.5)
	parser.add_argument("--cfg", dest = "cfgfile", help = "CFG file for the darknet architecture", default = "./cfg/yolov3.cfg")
	parser.add_argument("--weights", dest = "weightsfile", help = "Path for the pretrained weights file", default = "./weights/yolov3.weights")
	parser.add_argument("--reso", dest = "reso", help = "Input resolution for the darknet architecture, increase to increase accuracy and decrease to increase speed", default = 416, type = str)
	parser.add_argument("--video", dest = "videofile", help = "Video file to run detection on // input 'default' to use your default capturing device :id ==0", type = str, default = "default")
	parser.add_argument("--classes", dest = "classes", help = "Path to the class files", default = "./data/coco.names")
	return parser.parse_args()

def run(args):
	
	batch_size = int(args.bs)
	confidence = float(args.confidence)
	nms_thresh = float(args.nms_thresh)
	CUDA = torch.cuda.is_available() 

	num_classes = 80 
	classes = load_classes(args.classes)

	#set up the neural network 
	print("Loading the neural network....")
	model = Darknet(args.cfgfile)
	model.load_weights(args.weightsfile)
	print("Network successfully loaded")

	model.net_info["height"] = args.reso
	inp_dim = int(model.net_info["height"])
	assert inp_dim % 32 == 0 
	assert inp_dim > 32 

	#### IF GPU is available, use it 
	if CUDA: 
		model.cuda()

	#set the model in evaluation mode 
	model.eval()

	def write(x, results):
		c1 = tuple(x[1:3].int())
		c2 = tuple(x[3:5].int())
		img = results
		cls = int(x[-1])
		color = random.choice(colors)
		label = f"{classes[cls]}"
		cv2.rectangle(img, c1,c2,color,1)
		t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1,2)[0]
		c2 = c1[0] + t_size[0] + 3, c1[1]+t_size[1]+4
		cv2.rectangle(img,c1,c2,color,-1)
		cv2.putText(img, label, (c1[0],c1[1]+ t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 1,[225,255,255],1 )
		return img

	### Detection phase 
	videofile = args.videofile 
	if(videofile.lower().startswith("default")):
		videofile = 0

	cap = cv2.VideoCapture(videofile)

	assert cap.isOpened(), "Cannot capture source"

	frames = 0 
	start = time.time()

	while cap.isOpened():
		ret,frame = cap.read() 
		if ret:
			img = prep_image(frame,inp_dim)
			cv2.imshow("Real Frame", frame)
			im_dim = frame.shape[1], frame.shape[0]
			im_dim = torch.FloatTensor(im_dim).repeat(1,2)

			if CUDA:
				im_dim = im_dim.cuda() 
				img = img.cuda() 
			with torch.no_grad():
				output = model(Variable(img), CUDA)
			output = write_results(output,confidence, num_classes, nms_conf = nms_thresh)
			# print(type(output))
			if type(output) == int: 
				frames += 1 
				print("FPS : {:5.4 f}".format(frames/ (time.time()-start)))
				cv2.imshow("Frame",frame)
				key = cv2.waitKey(1)
				if key & 0xFF == ord('q'):
					break
				continue

			im_dim = im_dim.repeat(output.size(0),1)
			scaling_factor = torch.min(416/im_dim,1)[0].view(-1,1)

			output[:,[1,3]] -= (inp_dim - scaling_factor * im_dim[:,0].view(-1,1))/2
			output[:,[2,4]] -= (inp_dim - scaling_factor * im_dim[:,1].view(-1,1))/2

			output[:,1:5] /= scaling_factor

			for i in range (output.shape[0]):
				output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
				output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])

			colors = pickle.load(open("pallete","rb"))

			list(map(lambda x: write(x,frame),output))

			cv2.imshow("Detected Frame",frame)
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q'):
				print("Elapsed Time : {}".format(time.time()- start))
				break
			frames +=1 
			print("FPS : {:5.2f}".format(frames/(time.time()-start)))
		else:
			break



if __name__ == "__main__":
	args = arg_parse()
	run(args)
