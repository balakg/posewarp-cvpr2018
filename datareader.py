import numpy as np
import cv2
import scipy.io as sio
import os
import json

def getPersonScale(joints):
	torso_size = (-joints[0][1] + (joints[8][1] + joints[11][1])/2.0)
	peak_to_peak = np.ptp(joints,axis=0)[1]
	rcalf_size = np.sqrt((joints[9][1] - joints[10][1])**2 + (joints[9][0] - joints[10][0])**2)
	lcalf_size = np.sqrt((joints[12][1] - joints[13][1])**2 + (joints[12][0] - joints[13][0])**2)
	calf_size = (lcalf_size + rcalf_size)/2.0
	size = np.max([2.5*torso_size,5.0*calf_size,peak_to_peak*1.1]) 
	return (size/200.0)

def getExampleInfo(vid_name,frame_num,box,X):
	I_name_j = os.path.join(vid_name,str(frame_num+1)+'.jpg')
	joints = X[:,:,frame_num]-1.0
	box_j = box[frame_num,:]
	scale = getPersonScale(joints)
	pos = [(box_j[0] + box_j[2]/2.0), (box_j[1] + box_j[3]/2.0)] 
	lj = [I_name_j] + np.ndarray.tolist(joints.flatten()) + pos + [scale]
	return lj


def makeVidInfoList(vid_list_file):
	
	f = open(vid_list_file)
	vids = f.read().splitlines()
	f.close()	
	n_vids = len(vids)

	vid_info = []
		
	for vid_path in vids:

		path,vid_name = os.path.split(vid)
		info_name = path[:-6] + 'info/' + vid_name + '.mat'

		info = sio.loadmat(info_name)		
		box = info['data']['bbox'][0][0]
		X = info['data']['X'][0][0]

		#vid_info.append([box,X,vid_path])

		n_frames = X.shape[2]
		frames = np.random.choice(n_frames,2,replace=False)

		while(abs(frames[0] - frames[1])/(n_frames*1.0) <= 0.02):
			frames = np.random.choice(n_frames,2,replace=False)

		l = []
		l += getExampleInfo(vid_path,frames[0],box,X)
		l += getExampleInfo(vid_path,frames[1],box,X)
		l.append(class_id)
		ex.append(l)

	return ex


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

