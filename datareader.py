import numpy as np
import cv2
import scipy.io as sio
import os
from tensorflow.python.framework import ops

def makeWarpExampleList(param,seq_len=3):

	vid_pth = param['vid_pth']
	info_pth = param['info_pth']
	n_test_vids = param['n_test_vids']
	img_sfx = param['img_sfx']
	n_train_examples = param['n_train_examples']
	n_test_examples = param['n_test_examples']
	test_vids = param['test_vids']

	vid_names = [each for each in os.listdir(vid_pth)
                if os.path.isdir(os.path.join(vid_pth,each))]

	n_vids = len(vid_names)

	if(test_vids):
		test_vids = test_vids
		train_vids = list(set(range(n_vids)) - set(test_vids))	
	else:
		np.random.seed(17)
		random_order = np.random.permutation(n_vids).tolist()
		test_vids = random_order[0:n_test_vids]
		train_vids = random_order[n_test_vids:]

	ex_train = []
	ex_test = []

	for i in xrange(n_train_examples+n_test_examples):

		if(i < n_test_examples):
			vid = test_vids[np.random.randint(0,n_test_vids)]	
		else:
			vid = train_vids[np.random.randint(0,n_vids-n_test_vids)]

		vid_name = vid_names[vid]
		info = sio.loadmat(os.path.join(info_pth, vid_name + '.mat'))		
		box = info['data']['bbox'][0][0]
		X = info['data']['X'][0][0]

		n_frames = X.shape[2]-1 #last frames are sometimes bad

		'''	
		if(entire_vid):
			frames = np.arange(n_frames)
			frames.sort()
		else:	
		'''	
		
		frames = np.random.choice(n_frames,seq_len)
		if(np.random.rand() < 0.5):
			frames.sort()
		else:		
			frames[::-1].sort()		

		l = []
		for j in xrange(len(frames)):
			I_name_j = os.path.join(vid_pth,vid_name,str(frames[j]+1)+img_sfx)
			P_j = X[:,:,frames[j]].flatten()-1.0	
			box_j = box[frames[j],:]
			l += [I_name_j] + np.ndarray.tolist(P_j) + np.ndarray.tolist(box_j)

		if(i < n_test_examples):
			ex_test.append(l)
		else:	
			ex_train.append(l)

	return ex_train,ex_test
