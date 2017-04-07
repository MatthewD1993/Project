import argparse
import os.path
import caffe
import math
import cv2
import time
import numpy as np
import os.path
import matplotlib.image as mpimage
import matplotlib.pyplot as plt 



class TestData:
	'Test data handler'
	def __init__(self, test_txt, path_prefix='', tolerance=1.0):
		self.path_prefix = path_prefix
		self.test_txt =  test_txt
		self.tolerance = tolerance
		# print self.test_txt
		self.test_data_dict = []
		# self.summary = {}
		# self.error_range = 0 # Declare error_range existance
		self.num_items = 0
	def init(self):
		# Find head, left & right shoulders
		head = 0
		ls = 6
		rs = 5
		try:
			with open(self.test_txt,'r') as f:
				
				for line in f:
					self.num_items = self.num_items + 1
					entry = line.split()
					image_name	= self.path_prefix + '/' + entry[0]
					pos 		= np.asarray(entry[1].split(',')).reshape([7,2]).astype(np.float) # pos is a 7*2 2D array
					error_range = (np.linalg.norm(pos[head,:]-pos[ls,:]) + np.linalg.norm(pos[head,:]-pos[ls,:]))*0.5*self.tolerance
					# print error_range
					self.test_data_dict.append( (image_name, pos, error_range) )
					# self.test_data_dict[image_name][1] = 
			print self.num_items
			if self.num_items != 2381:
				print "List sorting error: missing!"
		except IOError as e:
			print "I/O error({0}): {1}".format(e.errno, e.strerror)
		except:
			print "Unexpected error."
			raise

	# def get_pos(self,key):
	# 	return self.test_data_dict[key][0]
	# def cal_error_range(self,pos):
	# 	return (np.linalg.norm(pos[head,:]-pos[ls,:]) + np.linalg.norm(pos[head,:]-pos[ls,:]))*0.5*self.tolerance
	def update(self, entry,evals):
		# self.summary[key] = (self.test_data_dict[key][1],evals)
		line = entry[0] + ' ' + str(entry[2]) + ' ' + ','.join(map(str,evals*1))
		return line
		# self.summary[key][1] = evals


# class Summary:
# 	'Summary of model accuracy'

def preprocess(img):
	[h,w] = img.shape[0:2]
	ratio = max(h,w)/256.0
	hor_padding = h >= w
	pad = abs(h-w)
	if h >= w:
	    im_square = cv2.copyMakeBorder(img,0,0,pad/2,pad-pad/2,cv2.BORDER_CONSTANT,value=[0,0,0])
	else:
	    im_square = cv2.copyMakeBorder(img,pad/2,pad-pad/2,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
	if max(h,w)!=256:
		im_square = cv2.resize(im_square, (256,256),cv2.INTER_AREA)
	return [im_square,ratio,hor_padding,pad]

def find_joints(heatmaps,ratio,hor_padding,pad):
	pred = np.zeros([7,2])
	for i in range(7):
		[x,y] = np.unravel_index(heatmaps[i,:,:].argmax(),[64,64])
		pred[i,:] = [y,x]
	# pred = find(heatmaps) # Find heatmap max coordinates in 64*64*7 matrix, pred is 7*2, (x,y)

	pred = pred*4
	if ratio != 1:
		pred = pred*ratio
	if hor_padding == True:
		pred[:,0] = pred[:,0] - pad/2  # Translate the coordinates by number of padding pixels
	else:
		print "Error: image width > height"
	return pred
 


	

def main():
	parser = argparse.ArgumentParser(description = "Benchmark Heatmap model accuray.")
	parser.add_argument("test_file",help = "The text file including image location and ground truth joint positions.")
	parser.add_argument("model",help = "The CNN model file.")
	parser.add_argument("weights",help = "The trained weights file.")
	parser.add_argument("prediction_file",help = "The path of output prediction file.")
	parser.add_argument("-p","--path",help = "The root path the images.", default ='')
	parser.add_argument("-e","--error_tolerance",type = float, 
		help = "The error tolerance. Default is 1, means average half distance from head to shoulders.",default = 1.0)
	args 		= parser.parse_args()
	path = args.path
	tolerance = args.error_tolerance
	prediction_file = args.prediction_file

	num_joints = 7

	# Initialize caffe prediction engine
	caffe.set_mode_gpu()
	caffe.set_device(0)
	net = caffe.Net(args.model, args.weights, caffe.TEST)

	print args.test_file
	time_obj_init = time.time()
	test_data = TestData(args.test_file, path, tolerance)
	test_data.init()
	end_obj_init = time.time()
	print "Time to initialize test data object {}".format(end_obj_init- time_obj_init)

	# if os.path.isfile('summary.txt'):
	summary = open('summary.csv','w+')
	prediction = open(prediction_file,'w')

	i = 0
	time_start_iter = time.time()
	for k in test_data.test_data_dict:
		image_raw = mpimage.imread(k[0])
		# Padding image to 256*256
		[image, ratio, hor_padding, pad] = preprocess(image_raw)

		# Feed image to caffe prediction engine
		net.blobs['data'].data[0,...] = np.transpose(image,(2,0,1)) # Transpose to chanle*weight*height
		result = net.forward()
		# print result['conv8'].shape
		heatmaps = result['conv5_fusion'][0,...] 

		# pred_joints and true_joints are 7*2 matrix
		pred_joints = find_joints(heatmaps,ratio,hor_padding,pad)
		true_joints = k[2]
		
		error 		= np.linalg.norm(np.transpose(pred_joints - true_joints), axis=0)
		evaluate 	= error < k[2]
		
		# print evaluate
		# plt.clf()
		# plt.imshow(image_raw)
		# plt.scatter(true_joints[:,0], true_joints[:,1], s=100, c='g', alpha=0.3)
		# plt.scatter(pred_joints[:,0], pred_joints[:,1], s=100, c='r', alpha=0.3)
		# plt.pause(1)
		# raw_input("Input Enter to continue...")


		line = test_data.update(k,evaluate)
		# update(universe,)
		summary.write(line+'\n')
		prediction.write(k[0] + ' '+ ','.join(map(str,pred_joints.flatten())) + '\n')
		i = i+1
		if not (i%100):
			now = time.time()
			print "Loop {0} items, time: {1}".format(i, now-time_start_iter)




	summary.close()
	prediction.close()

	if i!=2381:
		print "Error: missing!"
	print "Done!"


if __name__ =="__main__":
	main()