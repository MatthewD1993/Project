import argparse
import caffe
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from math import floor
from os import listdir

# Input image of any size, output the final conv5 layer 64*64*8 matrix
def preprocessImg(img,opt):
	[h,w] = img.shape[0:2]
	pad = abs(h-w)
	if h>w:
	    im_square = cv2.copyMakeBorder(img,0,0,pad/2,pad-pad/2,cv2.BORDER_CONSTANT,value=[0,0,0])
	else:
	    im_square = cv2.copyMakeBorder(img,pad/2,pad-pad/2,0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
	
	im_ready = cv2.resize(im_square, opt.dims)	
	# print im_ready.shape
	return im_ready

def applyNetImage(img, net, opt):
	#net.forward(img)
	im_processed = np.transpose(img,(2,0,1))
	net.blobs['data'].data[0,...] = im_processed
	params = net.forward()
	heatmaps = params['conv5_fusion'][0,...]
	# print 'Hello, I am fine'
	return heatmaps

def heatmapVisualize(heatmaps,img,opt):
	colors = np.array([[0,0,1],
	[0,1,0],
	[1,0,0],
	[1,1,0],
	[0,1,1],	
	[1,0,1],
	[0,0,0]])

	# print img.shape

	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	background = ~cv2.Canny(gray,80,120)
	background = np.tile(background[:,:,np.newaxis],(1,1,3))
	# print "background size is {}".format(np.amax(background))
	# print 'Hello, I am fine'
	# plt.imshow(background)
	# plt.show()
	# print background, background.shape
	clrmask = np.zeros((256,256,3))

	for i in xrange(7):
		alpha = np.divide(heatmaps[i,:,:],np.amax(heatmaps[i,:,:]))
		idx = alpha < 0.99
		alpha[idx] = 0.
		alpha_large = cv2.resize(alpha,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)
		# print  "alpha size is:{}".format(alpha_large.shape)
		# plt.imshow(alpha_large,cmap='gray',vmin=0,vmax=1)
		# plt.title("alpha large")
		# plt.show()
		alpha_ready = np.tile(alpha_large[:,:,np.newaxis],(1,1,3))
		# plt.imshow(alpha_ready[:,:,1])
		# plt.title("alpha ready")
		# plt.show()
		# print  "alpha size is:{}".format(alpha_ready.shape)
		# print type(colors[i,:].reshape((1,1,3))), colors[i,:].reshape((1,1,3)).shape
		color = colors[i,:].reshape((1,1,3))

		clr = np.multiply(255,np.tile(color,opt.dims+(1,)))
		# print clr.shape
		# print clr[:,:,2]
		# np.multiply(alpha_ready,clr)
		# print type(alpha_ready[1,1,1]),type(clr[1,1,1])
		# background = 0.3*np.multiply(alpha_ready,clr) + 0.7*np.multiply((1.- alpha_ready),background) 
		clrmask = clrmask + np.multiply(clr,alpha_ready)
		# background = 0.3*np.multiply(alpha_ready,clr) + 0.7*np.multiply((1.- alpha_ready),background) 
	idx = clrmask>255
	clrmask[idx] = 255
		
	# plt.imshow(background)
	# plt.imshow(clrmask)
	# plt.title("haha")
	# plt.show()

	pdf_img = 0.7*background + 0.3*clrmask
		# plt.imshow(np.multiply(np.subtract(1.,alpha_ready),background),cmap='gray',vmin=0,vmax=255)
		# plt.title('colors')
		# plt.show()
		# print "---------"
		# print np.multiply(np.subtract(1.,alpha_ready),background)
		# print "---------"
		# print pdf_img
	return pdf_img




class CaffeOptions:
	visualise = True
	useGPU = False
	dims = (256,256)
	numJoints = 7
	layerName = 'conv5_fusion'
	modelDefFile = '../matlab.prototxt'
	modelWeights = '../caffe-heatmap-flic.caffemodel'
	inputDirs = 'sample_images/'

def main():
	parser = argparse.ArgumentParser(
		description = 'Find human joints position from images')
	

	opt = CaffeOptions()
	#set up caffe
	caffe.set_mode_gpu()
	gpu_id = 0
	caffe.set_device(gpu_id);
	net = caffe.Net(opt.modelDefFile, opt.modelWeights, caffe.TEST)


	images = listdir("sample_images")
	# imagesa= images.sort()
	# print images,imagesa
	# a = './sample_images/'+images[0]
	# print a
	for im in images:
		img = cv2.imread( './sample_images/27.png')
		# print img
		im_square = preprocessImg(img,opt)
		# cv2.imshow('im_square', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		heatmaps = applyNetImage(im_square,net,opt)
		if opt.visualise == True:
			pdf_img = heatmapVisualize(heatmaps,im_square,opt)
			plt.imshow(pdf_img)
			plt.show()
		# cv2.imshow(pdf_img)
		# cv2.waitKey(0)	
		# cv2.destroyAllWindows()


	# test = heatmaps[0,:,:]
	# min = np.min(test)
	# max = np.max(test)
	# ok = np.floor((test-min)/(max-min)*255)
	# ok = ok.astype(int)
	# print test.shape
	# print ok
	# plt.figure(1)
	# plt.imshow(ok)
	# plt.show()


	

if __name__ == '__main__':
		main()



# function joints = applyNetImage(imgFile, net, opt)

# % Read & reformat input image
# img = imread(imgFile);
# input_data = prepareImagePose(img, opt);

# % Forward pass
# tic
# net.forward({input_data});
# features = net.blobs(opt.layerName).get_data();
# [joints, heatmaps] = processHeatmap(features, opt);
# disp(toc); 

# % Visualisation
# if opt.visualise
#     % Heatmap
#     heatmapVis = getConfidenceImage(heatmaps, img);
#     figure(2),imshow(heatmapVis);

#     % Original image overlaid with joints
#     figure(1),imshow(uint8(img));
#     hold on
#     plotSkeleton(joints, [], []);
#     hold off
# end
