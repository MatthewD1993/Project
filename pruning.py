import caffe
import sys
import os
import numpy as np

help_ = '''
Usage:
	pruning.py <net.prototxt> <net.caffemodel>
	Try range of pruning intensity, plot accurary diagrams with respect to training and testing dataset.

'''
if len(sys.argv) != 3:
	print help_
	sys.exit()
else:
	prototxt = sys.argv[1]
	weights = sys.argv[2]


caffe.set_mode_cpu()

net = caffe.Net(prototxt, weights, caffe.TEST)


'''
Step 1: Get a bunch of caffemodel, with different truncate level 
'''
# Only prune convolutional layers 
layers = [layer for layer in net.params.keys() ]
targets = filter(lambda layer: 'conv' in layer, layers)
# print targets



conv_params_size = [net.params[target][0].data.size for target in targets]
conv_bias_size = [net.params[target][1].data.size for target in targets]
print len(conv_bias_size)
print conv_params_size
print conv_bias_size

for truncate_level in np.linspace(0.1,2,0.1):

	for target in targets:
		data 	= net.params[target][0].data
		# print data
		shape	= data.shape
		mean	= data.mean()
		std		= data.std()
		trunc_range	= truncate_level*std
		idx 	= np.abs(data) < trunc_range
		# print idx
		net.params[target][0].data[idx] = 0
		# print net.params[target][0].data
		# raw_input("Please Enter:")
		# print shape
		# print type(net.params[target][0].data)
	model_path = 'prunedmodels/pruned_'+ str(truncate_level)+'.caffemodel'
	net.save(model_path)
''' 
Step 2: Measure the accuracy of each one, get the prediction file
'''

'''
Step 3: Plot truncated predictions with base predictions over 7 joints, plot 7 diagram.
'''