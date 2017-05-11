import numpy as np
import json
import cv2
import scipy.io as sio
import preprocess

n_joints = 14

def readMPIIStyleData(json_path,permute=True):
	data = []
	with open(json_path) as data_file:
		data = json.load(data_file)
		data = data['root']
		num_examples = len(data)

	if permute:
		np.random.seed(17)
		random_order = np.random.permutation(num_examples).tolist()
	else:
		random_order = np.arange(0,num_examples)

	examples = []

	mpii2cpm = [9,8,12,11,10,13,14,15,2,1,0,3,4,5]
	
	for i in xrange(num_examples):
		ri = random_order[i]

		img_name = str(data[ri]['img_name'])
		scale = data[ri]['scale_provided']

		if 'idx' in data[ri].keys():
			idx = data[ri]['idx']
		else:
			idx = 0
		obj_pos = np.asarray(data[ri]['objpos'])-1
		obj_pos = obj_pos.flatten().tolist()
		joints_cpm = np.zeros((n_joints,2))

		if 'joint_self' in data[ri].keys():
			joints = np.asarray(data[ri]['joint_self'])	
			for j in xrange(n_joints):
				joints_cpm[j,:] = joints[mpii2cpm[j],0:2]-1	
		
		joints_cpm = joints_cpm.flatten('F').tolist()
		example = [img_name,scale] + joints_cpm + obj_pos + [idx]
		examples.append(example)	

	return examples

