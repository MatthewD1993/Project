import numpy as np 
import matplotlib.pyplot as plt
import os.path
import matplotlib.image as mpimage
# how
num_joints = 7
def badpred(arr, tolerance):
	pred = map(int, arr[2].split(','))
	# Good prediction is 1, bad is 0
	if sum(pred) < (num_joints-tolerance):
		return True
def main():

	tolerance  = int(raw_input("Tolerance [0,7] is (0 means perfect):"))
	file = open('/mocapdata/data/flic/test_shuffle.txt','r')
	lines = file.readlines()
	predictions = {}

	with open('prediction.txt','r') as p:
		for line in p:
			entry = line.split()
			predictions[entry[0]] = map(float,entry[1].split(','))
	with open('summary.csv','r') as s:
		i = 0
		for line in s:
			arr = line.split()
			if badpred(arr,tolerance):
				img = mpimage.imread(arr[0])
				pred = np.asarray(predictions[arr[0]]).reshape([7,2])
				truth = np.asarray(map(float,lines[i].split()[1].split(','))).reshape([7,2])
				plt.clf()
				plt.imshow(img)
				plt.scatter(pred[:,0], pred[:,1], s=100, c = 'r', alpha = 0.5)
				plt.scatter(truth[:,0], truth[:,1], s=100, c = 'g', alpha = 0.5)
				plt.title("Number {} element {}".format(i,arr[0]))
				plt.pause(1)
				raw_input("Press Enter to continue...")

			i = i+1
		print i
		# print len(predictions)
if __name__ == "__main__":
	main()
