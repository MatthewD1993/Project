import caffe
import numpy as np 
import matplotlib.pyplot as plt

caffe.set_mode_cpu()

net = caffe.Net('deploy.prototxt','heatmap-xx.caffemodel',caffe.TEST)
print 'Hello world'
print net.params['conv1'][0].data.shape

print [(k,v[0].data.shape) for k,v in net.params.items()] 


for k,v in net.params.items():
	# graph name k, distr of v[0]
	t = v[0].data.flatten()
	i = 1
	fig = plt.figure()
	plt.title(k)
	plt.ylabel("Frequency")
	plt.hist(t,bins=50)
	#plt.show()
	fig.savefig('visualize_qw/'+k+'.png')
	i = i+1
	fig.clear()

# x = net.params["conv6"][0].data.flatten()
# print x.shape
# plt.hist(x,bins=50)
# plt.show()