import numpy as np
import cv2
import scipy.io as sio
import os
#from tensorflow.python.framework import ops
pennaction2cpm = [1, 1, 2, 4, 6, 3, 5, 7, 8, 10, 12, 9, 11, 13]
n_cpm_joints = 14
n_pa_joints = 13

def makePAWarpExampleList(param,n_train_examples,n_test_examples,actionNames=None,okVidsDir='good_pa_vids', seq_len=2):

	vid_pth = param['vid_pth']
	info_pth = param['info_pth']
	img_sfx = param['img_sfx']

	# load all available action classes by default
	if actionNames is None or actionNames == 'all':
		actionNames = ['baseball_pitch','baseball_swing','jump_rope','jumping_jacks','tennis_forehand','tennis_serve']

	vid_names = []
	for an in actionNames:
		okVidsFile = os.path.join( okVidsDir, an + '.txt' )
		with open(okVidsFile) as f:
			content = f.readlines()
			vid_names += [x.strip() for x in content]

	n_vids = len(vid_names)

	ex_train = []
	ex_test = []

	# load train and test sets according to PA split
	test_vids = []
	train_vids = []

	# go through all OK vids to see which ones are specified as test vs train
	for i in range(n_vids):
		vid_name = vid_names[i]
		info = sio.loadmat(os.path.join(info_pth, vid_name + '.mat'))
		isTrain = info['train'][0][0] == 1
		if isTrain:
			train_vids.append(i)
		else:
			test_vids.append(i)

	for i in range(n_train_examples+n_test_examples):
		if(i < n_test_examples):
			vidId = test_vids[np.random.randint(0,len(test_vids))]
		else:
			vidId = train_vids[np.random.randint(0,len(train_vids))]

		vid_name = vid_names[vidId]
		info = sio.loadmat(os.path.join(info_pth, vid_name + '.mat'))
		isTrain = info['train'][0][0]==1
		n_frames = info['x'].shape[0]-1

		joints = np.concatenate((np.reshape(info['x'][:n_frames, :], (n_frames,n_pa_joints, 1)), 
				 np.reshape(info['y'][:n_frames, :], (n_frames,n_pa_joints, 1))), axis=2)

		joints_cpm = np.zeros((n_frames,n_cpm_joints, 2))
		for j in range(0, n_cpm_joints):
			joints_cpm[:,j,:] = joints[:,pennaction2cpm[j] - 1, 0:2]
		joints_cpm[:,1,:] = (joints_cpm[:,2,:] + joints_cpm[:,5,:])/2.0
		#headLen = np.linalg.norm( joints_cpm[:,0,:] - joints_cpm[:,1,:],axis=2)
		headVec = joints_cpm[:,0,:] - joints_cpm[:,1,:]
		#print(headVec)

		joints_cpm[:,0,:] = joints_cpm[:,0,:] + np.multiply(headVec,0.3)
		joints_cpm[:, 1, :] = joints_cpm[:, 1, :] + np.multiply(headVec, 0.3)
		box = info['bbox'][:n_frames,:]

		frames = np.random.choice(n_frames,seq_len)
		if(np.random.rand() < 0.5):
			frames.sort()
		else:		
			frames[::-1].sort()		

		l = []
		for j in range(len(frames)):
			I_name_j = os.path.join(vid_pth,vid_name,'{:06.0f}'.format(frames[j]+1)+img_sfx)
			P_j = joints_cpm[frames[j],:,:].flatten()-1.0
			box_j = box[frames[j],:]
			rShinLen = np.linalg.norm(joints_cpm[frames[j],9,:]-joints_cpm[frames[j],10,:])
			lShinLen = np.linalg.norm(joints_cpm[frames[j],12,:]-joints_cpm[frames[j],13,:])

			# center of bbox
			pos = [(box_j[0] + box_j[2])/2.0, (box_j[1] + box_j[3])/2.0]

			bboxH = box_j[3]-box_j[1]
			personH = max( np.linalg.norm(joints_cpm[frames[j],0,:]-joints_cpm[frames[j],10,:]),
										 np.linalg.norm(joints_cpm[frames[j],0,:]-joints_cpm[frames[j],13,:]),
										 np.linalg.norm(joints_cpm[frames[j], 4, :] - joints_cpm[frames[j], 10, :]),
										 np.linalg.norm(joints_cpm[frames[j], 7, :] - joints_cpm[frames[j], 13, :]) )

			#if bboxH > cv2.imread(I_name_j).shape[0]*0.9: # if bounding box is too large compared to frame, use shin instead
			if bboxH > 1.2 * personH or int(bboxH)==0:
				scale = max(rShinLen*4.5, lShinLen*4.5)/200.0
			else:
				scale = bboxH/200.0

			if(scale <= 0):
				print('{},{}'.format(scale,bboxH))


			l += [I_name_j] + np.ndarray.tolist(P_j) + pos + [scale]

		if not isTrain:
			ex_test.append(l)
		else:	
			ex_train.append(l)

	return ex_train,ex_test


if __name__=='__main__':
	testParams = { 'vid_pth':'..\\..\\Matlab\\Datasets\\Penn_Action\\Penn_Action\\frames',
	'info_pth': '..\\..\\Matlab\\Datasets\\Penn_Action\\Penn_Action\\labels',
	'img_sfx': '.jpg',
	}

	segLen = 2
	exLen = 1 + 14*2 + 2 + 1

	ex_train, ex_test = makePAWarpExampleList(testParams,2000,0,seq_len = segLen)

	# write preview jpgs of joints, obj_pos and scales
	imCount = 0
	ims = None
	setCount = 0
	imsPerCol = 8
	for ex in ex_train+ex_test:
		segIms = None
		for t in range(segLen):
			imFile = ex[t*exLen+0]
			vidNum = os.path.basename(os.path.dirname(imFile))
			scale = ex[t * exLen + 1 + 14 *2 + 2]
			pos = [int(i) for i in ex[t * exLen + 1 + 14 *2:t * exLen + 1 + 14 *2+2]]
			joints = np.reshape([int(i) for i in ex[t*exLen+1:t*exLen+1+14*2]],(14,2))
			print(scale)
			im = cv2.line( cv2.drawMarker(cv2.imread( imFile ), tuple(pos), color=(255,0,0)), \
				(pos[0],int(pos[1]-scale*100.0)), (pos[0],int(pos[1]+scale*100.0)), color=(0,0,255), thickness=2)
			for i in range(14):
				im = cv2.drawMarker( im, tuple(joints[i,:]), color=(0,255,0) )

			isTrain = ex in ex_train
			im = cv2.resize(im, dsize=(320,240))
			if segIms is None:
				im = cv2.putText(im,'Vid {}, train={}'.format(vidNum,isTrain), (20,30), fontFace=cv2.FONT_HERSHEY_PLAIN,fontScale=1,color=(0,255,0))
				segIms = im
			else:
				segIms = np.concatenate((segIms,im),axis=0)

		if ims is None:
			ims = segIms
			imCount += 1
		elif imCount < imsPerCol:
			ims = np.concatenate((ims,segIms),axis=1)
			imCount+=1
		elif imCount >= imsPerCol:
			imCount = 0
			cv2.imwrite('pa_preview\\pa_set_{}.jpg'.format(setCount), ims)
			setCount += 1
			ims = None

	cv2.waitKey(0)

