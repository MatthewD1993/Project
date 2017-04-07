import numpy as np 
with open('/mocapdata/cdeng/mpii/train_shuffle.txt','r') as f, open('train_shuffle.txt','w') as new_train:
	for line in f:
		[name, pos] = line.split()
		cut_pos = ','.join(pos.split(',')[0:14])
		new_train.write(name+' ' + cut_pos + ' ' + '0,0,0,0,0' + ' ' + '0' +'\n')

with open('/mocapdata/cdeng/mpii/test_shuffle.txt','r') as f, open('test_shuffle.txt','w') as new_train:
	for line in f:
		[name, pos] = line.split()
		cut_pos = ','.join(pos.split(',')[0:14])
		new_train.write(name+' ' + cut_pos +  ' ' + '0,0,0,0,0' + ' ' + '0' +'\n')