import numpy as np
import cv2
import scipy.io as sio
import os
import json


def makeVidInfoList(vid_list_file):
	
	f = open(vid_list_file)
	vids = f.read().splitlines()
	f.close()	
	n_vids = len(vids)

	vid_info = []
	
	for i in range(n_vids):

		path,vid_name = os.path.split(vids[i])
		info_name = path[:-6] + 'info/' + vid_name + '.mat'

		info = sio.loadmat(info_name)		
		box = info['data']['bbox'][0][0]
		X = info['data']['X'][0][0]

		vid_info.append([info,box,X,vids[i]])

		'''
		n_frames = X.shape[2]
		frames = np.random.choice(n_frames,2,replace=False)

		while(abs(frames[0] - frames[1])/(n_frames*1.0) <= 0.02):
			frames = np.random.choice(n_frames,2,replace=False)

		l = []
		l += getExampleInfo(vid_path,frames[0],box,X)
		l += getExampleInfo(vid_path,frames[1],box,X)
		#l.append(class_id)
		ex.append(l)
		'''
	print len(vid_info)

	return vid_info


'''
def makeWarpExampleList(vid_file,n_examples):
	
	f = open(vid_file)
	vid_lines = f.read().splitlines()
	f.close()
	
	n_vids = len(vid_lines)

	ex = []
	for i in xrange(n_examples):
		vid_line = vid_lines[np.random.randint(0,n_vids)]
		vid_path = vid_line[0:-2]
		class_id = vid_line[-1]

		path,vid_name = os.path.split(vid_path)
		info_name = path[:-6] + 'info/' + vid_name + '.mat'

		info = sio.loadmat(info_name)		
		box = info['data']['bbox'][0][0]
		X = info['data']['X'][0][0]

		n_frames = X.shape[2]
		frames = np.random.choice(n_frames,2,replace=False)

		#print i,n_examples
		while(abs(frames[0] - frames[1])/(n_frames*1.0) <= 0.02):
			frames = np.random.choice(n_frames,2,replace=False)

		l = []
		for j in xrange(len(frames)):
			l += getExampleInfo(vid_path,frames[j],box,X)
		
		l.append(class_id)

		ex.append(l)

	return ex
'''


def makeActionExampleList(vid_list_file,example_num):
	
	f = open(vid_list_file)
	vid_paths = f.readlines()
	f.close()
	
	vid_path = vid_paths[example_num]
	path,vid_name = os.path.split(vid_path)	
	info_name = path[:-6] + 'info/' + vid_name[:-1] + '.mat'

	info = sio.loadmat(info_name)		
	box = info['data']['bbox'][0][0]
	X = info['data']['X'][0][0]

	ex = []
	for i in xrange(X.shape[2]):
		l = []
		l += getExampleInfo(vid_path,0,box,X)
		l += getExampleInfo(vid_path,i,box,X)
		ex.append(l)

	return ex

