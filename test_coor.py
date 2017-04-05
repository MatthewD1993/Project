import matplotlib.pyplot as plt 
import matplotlib.image as mpimage
import numpy as np 
import os.path
import sys
test_file = "/mocapdata/cdeng/mpii/test_shuffle.txt"

num = 0

with open(test_file,'r') as f:
	for line in f:
		details = line.split()
		img_pos = "/mocapdata/cdeng/mpii/" + details[0]
		try:
			img = mpimage.imread(img_pos)
			print img.shape
			
			joints = np.asarray(details[1].split(',')).reshape([14,2]).astype(np.float)
			# positions = joints[6,:]
			# print positions
			for i in range(7):
				plt.clf()
				plt.imshow(img)
				plt.scatter(joints[i,0],joints[i,1], s=100, c='g', alpha=0.5)
				plt.title(details[0])
				plt.pause(1)
				cmd = raw_input('Please enter:')
				# print cmd
				# print cmd=='q'
				# if cmd == 'q':
				# 	sys.exit(0)
			# print positions
		except:
			num = num +1
# 		if not os.path.isfile(img_pos):
# 			print img_pos
# 			num_missing = num_missing + 1 

print num

