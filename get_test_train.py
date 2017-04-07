# 
import numpy as np 

train_file = '/mocapdata/data/flic/train_shuffle.txt'
train_test = './train_test.txt'
with open(train_file,'r') as train:
	lines = train.readlines()
	# print lines
	np.random.shuffle(lines)
	new_file = lines[0:2381]
	# print new_file


with open(train_test, 'w') as test:
	for entry in new_file:
		test.write(entry )
with open(train_test,'r') as f:
	lines = f.readlines()
	print lines[0]