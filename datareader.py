import numpy as np
import cv2
import scipy.io as sio
import os
from tensorflow.python.framework import ops

def makeTransferExampleList(vid_pth,info_pth,n_test_vids,n_end_remove,img_sfx,n_train_examples,n_test_examples):

	vid_names = [each for each in os.listdir(vid_pth)
                if os.path.isdir(os.path.join(vid_pth,each))]

	n_vids = len(vid_names)
	np.random.seed(17)
	random_order = np.random.permutation(n_vids).tolist()
	test_vids = random_order[0:n_test_vids]
	train_vids = random_order[n_test_vids:]

	ex_train = []
	ex_test = []

	for i in xrange(n_train_examples+n_test_examples):

		#choose video
		if(i < n_test_examples):
			vid = test_vids[np.random.randint(0,n_test_vids)]	
		else:
			vid = train_vids[np.random.randint(0,n_vids-n_test_vids)]

		vid_name = vid_names[vid]
		info = sio.loadmat(os.path.join(info_pth, vid_name + '.mat'))		
		box = info['data']['bbox'][0][0]
		X = info['data']['X'][0][0]

		n_frames = X.shape[2]-n_end_remove #Some predictions at the end are bad
		
		#choose frames
		frame0 = np.random.randint(0,n_frames)
 		frame1 = np.random.randint(0,n_frames)
		while(frame1 == frame0): #frame1 < 0.85*frame0 or frame1 > 1.15*frame0):
			frame1 = np.random.randint(0,n_frames)

		I0_name = os.path.join(vid_pth,vid_name,str(frame0+1)+img_sfx)
		P0 = X[:,:,frame0].flatten()-1.0	
		box0 = box[frame0,:]
	

		I1_name = os.path.join(vid_pth,vid_name,str(frame1+1) + img_sfx)
		P1 = X[:,:,frame1]-1.0	
		box1 = box[frame1,:]

		l = [I0_name,I1_name] + np.ndarray.tolist(P0) + np.ndarray.tolist(P1.flatten())
		l += np.ndarray.tolist(box0) + np.ndarray.tolist(box1)

		if(i < n_test_examples):
			ex_test.append(l)
		else:	
			ex_train.append(l)

	return ex_train,ex_test



def makePoseExampleList(vid_pth,info_pth,n_test,n_end_remove,img_sfx):

	vid_names = [each for each in os.listdir(vid_pth)
                if os.path.isdir(os.path.join(vid_pth,each))]

	n_vids = len(vid_names)
	np.random.seed(17)
	random_order = np.random.permutation(n_vids).tolist()

	ex_train = []
	ex_test = []

	for i in xrange(n_vids):
		ri = random_order[i]
		vid_name = vid_names[ri]
		info = sio.loadmat(os.path.join(info_pth, vid_name + '.mat'))		
		box = info['data']['bbox'][0][0]
		X = info['data']['X'][0][0]

		n_frames = X.shape[2]

		for frame1 in xrange(0,n_frames-n_end_remove):
			I0_name = os.path.join(vid_pth,vid_name,str(frame1+1)+img_sfx)
			P0 = X[:,:,frame1].flatten()-1.0	
			box0 = box[frame1,:]
			
			l = [I0_name] + np.ndarray.tolist(P0)
			l += np.ndarray.tolist(box0)

			if(i < n_test):
				ex_test.append(l)
			else:	
				ex_train.append(l)

	return ex_train,ex_test	
