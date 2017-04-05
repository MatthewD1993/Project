import argparse
import numpy as np
import math
import matplotlib.pyplot as plt 
import matplotlib.image as mpimage


def main():
	parser = argparse.ArgumentParser(description = "Plot accuray diagrams with different error range.")
	parser.add_argument("pred_file")
	parser.add_argument("test_file")
	args = parser.parse_args()
	pred_file = args.pred_file
	test_file = args.test_file
	pred = open(pred_file,'r')
	preds = pred.readlines()
	test = open(test_file, 'r')
	truths = test.readlines()
	num_pred = len(preds)


	num_joints = 7
	if num_pred != len(truths):
		print "Num of test images and num of preds not matching!"
		return

	dist_table = np.zeros([num_pred, num_joints])
	for i in range( num_pred ):
		[name, temp,_,_] = truths[i].split()
		
		# Note, mpii dataset provide 14 joints positions
		jos = np.array(map(float, temp.split(',')))
		joints = jos.reshape(7,2)
		# This number is used for normalization.
		ref = 0.5*( np.linalg.norm(joints[0,:] - joints[5,:]) + np.linalg.norm(joints[0,:] - joints[6,:]) )
		if ref == 0:
			img = mpimage.imread("/mocapdata/cdeng/mpii/"+name)
			plt.imshow(img)
			plt.pause(1)
			raw_input("Any input:")
		else:
			pred_joints = np.array(map(float, preds[i].split()[1].split(',')))
			dist = np.linalg.norm(np.array(pred_joints-jos[0:14]).reshape([num_joints,2]), axis = 1)
			dist_norm = dist/ref
			dist_table[i] = dist_norm

	data = np.zeros([41,num_joints])
	with open('plot.csv','w') as p:
		for i,scale in enumerate(np.linspace(0,1,41)):
			data[i] = np.sum((dist_table <= scale)*1.0, 0)/num_pred*100
			p.write(','.join(data[i].astype(str)) + '\n')
		

	# for i in range(num_joints):
	# 	plt.plot()









	test.close()
	pred.close()
if __name__ == "__main__":
	main()