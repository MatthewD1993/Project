import os
from os.path import isfile, join



# List all caffemodel file, and process to get the iteration numbers
# Aplly those caffemodels to get prediction file for each one
# Analyse the prediction files to get plot.csv for each one. The rest visualize work can be done in Excel
caffe_root = '/home/cdeng/caffe-heatmap'
data_dir = os.path.join(caffe_root, 'data/snapshots' )
print data_dir
# caffemodels is in random order
caffemodels = [f for f in os.listdir(data_dir) if isfile( join(data_dir,f) ) ]
print caffemodels
for weight in caffemodels:
	i = weight.split('_')[3].split('.')[0]
	pred_file = 'Statistics/fine_'+ i + '_pred_flic.txt'
	test_file = '/mocapdata/data/flic/test_shuffle.txt'
	plot_file = 'Statistics/fine_'+ i +'_plot_flic.csv'
	cmd1 ='python benchmark.py test_shuffle.txt ../deploy.prototxt ' + join(data_dir,weight) + ' ' + pred_file + ' ' + '-p /mocapdata/data/flic'
	os.system(cmd1)
	cmd2 = 'python plot.py ' + pred_file + ' ' + test_file + ' ' + plot_file
	os.system(cmd2)


