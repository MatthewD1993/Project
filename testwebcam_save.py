import numpy as np
import caffe
import matplotlib.pyplot as plt
import cv2
import os
import time

# Set up camera input
cap = cv2.VideoCapture(0)
width = cap.get(3)
height = cap.get(4)
size = min(width, height)
_ = cap.set(3,size)
_ = cap.set(4,size)
dim = (256,256)

# Set up caffe model
caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net('../deploy.prototxt','../caffe-heatmap-flic.caffemodel',caffe.TEST)
heat_layer = 'conv5_fusion'

# OpenCV read in BGR format iamge!

# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)


count = 0
area = np.pi*10**2
# plt.ion()

if not os.path.exists("mymotion"):
	os.makedirs("mymotion")

try:
	while(True):
		begin = time.time()

		# Capture frame-by-frame
		count = count + 1
		ret, frame = cap.read()

		resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

		# Our operations on the frame come here
		#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


		# TODO Feed it to CNN
		# reshape input image to channel*weight*height
		net.blobs['data'].data[0,...] = np.transpose(resized,(2,0,1))

		b1 = time.time()
		params = net.forward()
		b2 = time.time()
		# Heatmap is 7*64*64
		heatmap_joints = params[heat_layer][0,...]
		heatmap_shape = heatmap_joints.shape[1:3]
		# Process heatmap
		heatmap = np.zeros((7,2))

		for i in range(7):
			[x,y] = np.unravel_index(heatmap_joints[i,...].argmax(),heatmap_shape)
			heatmap[i,:] = [y,x]
		b,g,r = cv2.split(resized)
		# fig = plt.figure()
		rgb = cv2.merge([r,g,b])
		plt.clf()
		plt.imshow(rgb)
		plt.scatter(4*heatmap[:,0],4*heatmap[:,1], s=area, c='r', alpha=0.3)
		# Pause # seconds
		plt.pause(0.1)

		# Display the resulting frameqq
		# cv2.imshow('frame',resized)




		cv2.imwrite("mymotion/%05d.jpg" % count, resized)     # save frame as JPEG file
		
		end = time.time()
		print('Time used for frame %d is %f seconds' % (count,end-begin))
except KeyboardInterrupt:
	pass
	# if cv2.waitKey(500) & 0xFF == ord('q'):
	# 	break

# When everything done, release the capture
cap.release()
# cv2.destroyAllWindows()
