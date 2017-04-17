import numpy as np
import json
import cv2
import scipy.io as sio
from scipy import interpolate
#import transformations

def putGaussianMaps(H,joints,sigma,stride):
	x = np.array(range(H.shape[1]))*stride
	y = np.array(range(H.shape[0]))*stride

	xv,yv = np.meshgrid(x,y,sparse=False,indexing='xy')
	
	for i in xrange(joints.shape[0]):	
		d = (xv - joints[i,0]) * (xv - joints[i,0]) + (yv - joints[i,1]) * (yv - joints[i,1])
		exponent = d/(2*sigma*sigma)
		H[:,:,i] = np.exp(-exponent)

def randScale( param ):
	rnd = np.random.rand()
	return ( param['scale_max']-param['scale_min']) * rnd + param['scale_min']

def randRot( param ):
	return (np.random.rand()-0.5)*2 * param['max_rotate_degree']

def randShift( param ):
	shift_px = param['max_px_shift']

	x_shift = int(shift_px * (np.random.rand()-0.5))
	y_shift = int(shift_px * (np.random.rand()-0.5))
	return x_shift, y_shift

def randSat(param):

	min_sat = 1 - param['max_sat_factor']
	max_sat = 1 + param['max_sat_factor']

	return np.random.rand()*(max_sat-min_sat) + min_sat

def makeMask(ctr,sigma_x,sigma_y,crop_size_x,crop_size_y):
	x = np.array(range(crop_size_x))
	y = np.array(range(crop_size_y))

	xv,yv = np.meshgrid(x,y,sparse=False,indexing='xy')
	d = ((xv - ctr[0]) * (xv - ctr[0]))/(2.0*sigma_x*sigma_x) + (
		(yv - ctr[1]) * (yv - ctr[1]))/(2.0*sigma_y*sigma_y)
	H = np.exp(-d)
	H = H/np.amax(H)
	return H

def augmentJoints(joints,f,limbs):
	for i in xrange(limbs.shape[0]):
		for j in xrange(1,f-1,1):
			xij = (f-j)/(f*1.0) * joints[limbs[i,0]-1,0] + j/(f*1.0) * joints[limbs[i,1]-1,0]
			yij = (f-j)/(f*1.0) * joints[limbs[i,0]-1,1] + j/(f*1.0) * joints[limbs[i,1]-1,1]
			joints = np.vstack((joints,[xij,yij]))


	#add center of body
	x = (joints[2,0]+joints[5,0]+joints[8,0]+joints[11,0])/4.0
	y = (joints[2,1]+joints[5,1]+joints[8,1]+joints[11,1])/4.0
	joints = np.vstack((joints,[x,y]))

	return joints


def makeInitialWarpField(joints0,joints1,sigma,crop_size_x,crop_size_y):
	limbs = np.array([[1,2],[3,4],[4,5],[6,7],[7,8],[9,10],[10,11],[12,13],[13,14],[3,9],[6,12],[3,6],[9,12]])	

	joints0 = augmentJoints(joints0,15,limbs)
	joints1 = augmentJoints(joints1,15,limbs)

	xv,yv = np.meshgrid(np.array(range(crop_size_x)),np.array(range(crop_size_y)),sparse=False,indexing='xy')

	V = np.zeros((crop_size_y,crop_size_x,2))
	M = np.zeros((crop_size_y,crop_size_x))

	for i in xrange(joints0.shape[0]):	
		vx = joints0[i,0] - joints1[i,0]
		vy = joints0[i,1] - joints1[i,1]

		d = (xv - joints1[i,0]) * (xv - joints1[i,0]) + (yv - joints1[i,1]) * (yv - joints1[i,1])
		
		exponent = np.exp(-d/(2*sigma*sigma))
		V[:,:,0] += vx * exponent
		V[:,:,1] += vy * exponent
		M += exponent

	M += 1e-5

	V[:,:,0] /= M
	V[:,:,1] /= M

	M = np.reshape(M,(crop_size_y,crop_size_x,1))

	return V


def makeJointWeightMask(joints,sigma,crop_size_x,crop_size_y):
	limbs = np.array([[1,2],[3,4],[4,5],[6,7],[7,8],[9,10],[10,11],[12,13],[13,14],[3,9],[6,12],[3,6],[9,12]])	

	joints = augmentJoints(joints,5,limbs)

	xv,yv = np.meshgrid(np.array(range(crop_size_x)),np.array(range(crop_size_y)),sparse=False,indexing='xy')

	H = np.zeros((crop_size_y,crop_size_x))

	for i in xrange(joints.shape[0]):	
		d = (xv - joints[i,0]) * (xv - joints[i,0]) + (yv - joints[i,1]) * (yv - joints[i,1])
		
		exponent = np.exp(-d/(2*sigma*sigma))
		exponent = exponent/np.amax(exponent)
		H = np.maximum(H,exponent)

	H = np.reshape(H,(crop_size_y,crop_size_x,1))

	return H

def transferExampleGenerator(examples,batch_size,param):
    
	crop_size_x = param['IMG_WIDTH']
	crop_size_y = param['IMG_HEIGHT']
	stride = param['posemap_downsample']
	sigma = param['sigma']
	n_joints = param['n_joints']
	target_dist = param['target_dist']	

	X_img = np.zeros((batch_size,crop_size_y,crop_size_x,3))
	X_pose = np.zeros((batch_size,crop_size_y/stride,crop_size_x/stride,n_joints*2))
	Y = np.zeros((batch_size,crop_size_y,crop_size_x,3))
	#V = np.zeros((batch_size,crop_size_y,crop_size_x,2))
	#M = np.zeros((batch_size,crop_size_y,crop_size_x,1))

	while True:
		for i in xrange(batch_size):
			example = examples[np.random.randint(0,len(examples))] 
			I0 = cv2.imread(example[0])
			I1 = cv2.imread(example[1])

			joints0 = np.reshape(np.array(example[2:30]), (14,2))
			joints1 = np.reshape(np.array(example[30:58]), (14,2))
			scale0 = example[61]/200.0
			scale1 = example[65]/200.0
			scale = np.max([scale0,scale1])
			pos0 = np.array([example[58] + example[60]/2.0, example[59] + example[61]/2.0])
			pos1 = np.array([example[62] + example[64]/2.0, example[63] + example[65]/2.0])

			rshift = randShift(param)
			rscale = randScale(param)
			rdegree = randRot(param)
			rsat = randSat(param)

			I0 = I0/255.0
			I1 = I1/255.0

			I0,joints0,pos0 = augScale(I0,scale,target_dist,rscale,joints0,pos0)
			I1,joints1,_ = augScale(I1,scale,target_dist,rscale,joints1)	

			I0,joints0,_ = augCrop(I0,crop_size_x,crop_size_y,rshift,joints0,pos0)
			I1,joints1,_ = augCrop(I1,crop_size_x,crop_size_y,rshift,joints1,pos0)	

			I0,joints0,_ = augRotate(I0,crop_size_x,crop_size_y,rdegree,joints0)
			I1,joints1,_ = augRotate(I1,crop_size_x,crop_size_y,rdegree,joints1)

			I0 = augSaturation(I0,rsat)
			I1 = augSaturation(I1,rsat)

			I0 = I0 - 0.5
			I1 = I1 - 0.5

			H0 = np.zeros((crop_size_y/stride, crop_size_x/stride,n_joints))
			putGaussianMaps(H0,joints0,sigma,stride)
	
			H1 = np.zeros((crop_size_y/stride, crop_size_x/stride,n_joints))
			putGaussianMaps(H1,joints1,sigma,stride)

			'''		
			#Make obj output heatmap
			max_vals = np.amax(joints1,axis=0)
			min_vals = np.amin(joints1,axis=0) 
			obj_ctr = (min_vals + max_vals)/2.0
			sigma_x = (max_vals[0]-min_vals[0])/2
			sigma_y = (max_vals[1]-min_vals[1])/2 
	
			mask = makeMask(obj_ctr,sigma_x,sigma_y,crop_size_x,crop_size_y)
			#M = cv2.resize(M,(0,0), fx=1.0/stride, fy=1.0/stride)
			M[i,:,:,:] = np.reshape(mask,(crop_size_y,crop_size_x,1))*2 - 1
			#M = np.reshape(M, (crop_size_y/stride,crop_size_x/stride,1))	
			'''

			#V[i,:,:,:] = makeInitialWarpField(joints0,joints1,2,crop_size_x,crop_size_y)

			X_img[i,:,:,:] = I0
			Y[i,:,:,:] = I1
			X_pose[i,:,:,:] = np.concatenate((H0,H1),axis=2)

		yield (X_img,X_pose,Y)


def poseExampleGenerator(examples,batch_size,param):
    
	crop_size_x = param['IMG_WIDTH']
	crop_size_y = param['IMG_HEIGHT']
	stride = param['posemap_downsample']
	sigma = param['sigma']
	n_joints = param['n_joints']
	
	X_img = np.zeros((batch_size,crop_size_y,crop_size_x,3))
	#X_pose = np.zeros((batch_size,crop_size_y/stride,crop_size_x/stride,n_joints*2))
	X_pose = np.zeros((batch_size, n_joints*2))

	while True:
		for i in xrange(batch_size):
			example = examples[np.random.randint(0,len(examples))] 

			I = cv2.imread(example[0])

			joints = np.reshape(np.array(example[1:29]), (14,2))
			scale = example[32]/200.0
			pos = np.array([example[29] + example[31]/2.0, example[30] + example[32]/2.0])

			rshift = randShift(10)
			rscale = randScale(param)

			I,joints,pos = augScale(I,scale,param,rscale,joints,pos)
			I,joints = augCrop(I,param,rshift,joints,pos)	

			I = (I-128.0)/255.0

			#H0 = np.zeros((crop_size_y/stride, crop_size_x/stride,n_joints))
			#putGaussianMaps(H0,joints0,sigma,stride)

			X_img[i,:,:,:] = I
			X_pose[i,:] = joints.flatten()

		yield (X_img,X_pose)



def augScale(I,obj_scale, target_dist, scale_rand, joints, obj_pos = None):
	scale_multiplier = scale_rand
	scale_abs = target_dist/obj_scale
	scale = scale_abs * scale_multiplier
	I = cv2.resize(I,(0,0),fx=scale,fy=scale)
	
	joints = np.copy(joints * scale)
	
	if(obj_pos is not None):
		obj_pos = np.copy(obj_pos * scale)

	return I,joints,obj_pos


def augRotate(I,crop_size_x,crop_size_y,degree_rand,joints,obj_pos=None):

	if degree_rand is not None:
		degree = degree_rand
	else:
		degree = randRot( param )
		print('Rot: {}'.format(degree))

	h = I.shape[0]
	w = I.shape[1]	

	center = ( (w-1.0)/2.0, (h-1.0)/2.0 )
	R = cv2.getRotationMatrix2D(center,degree,1)	
	I = cv2.warpAffine(I,R,(crop_size_x,crop_size_y))

	if(obj_pos is not None): 
		obj_pos = rotatePoint(obj_pos,R)

	joints = np.copy(joints)
	for i in xrange(joints.shape[0]):
		joints[i,:] = rotatePoint(joints[i,:],R)

	return I,joints,obj_pos

def rotatePoint(p,R):
	x_new = R[0,0] * p[0] + R[0,1] * p[1] + R[0,2]
	y_new = R[1,0] * p[0] + R[1,1] * p[1] + R[1,2]	
	return np.array((x_new,y_new))

def augCrop(I,crop_size_x,crop_size_y,rand_shift,joints,obj_pos):
	x_shift = rand_shift[0]
	y_shift = rand_shift[1]

	x_offset = (crop_size_x-1.0)/2.0 - (obj_pos[0] + x_shift)
	y_offset = (crop_size_y-1.0)/2.0 - (obj_pos[1] + y_shift)

	T = np.float32([[1,0,x_offset],[0,1,y_offset]])	
	I = cv2.warpAffine(I,T,(crop_size_x,crop_size_y))

	joints = np.copy(joints)
	obj_pos = np.copy(obj_pos)

	joints[:,0] += x_offset
	joints[:,1] += y_offset

	obj_pos[0] += x_offset
	obj_pos[1] += y_offset

	return I,joints,obj_pos


def augSaturation(I,rsat):
	I  *= rsat
	I[I > 1] = 1
	return I

'''
	limbs = np.array([[1,2],[3,4],[4,5],[6,7],[7,8],[9,10],[10,11],[12,13],[13,14],[3,9],[6,12],[3,6],[9,12]])	
	for i in xrange(limbs.shape[0]):
		for j in xrange(1,9,1):
			xij = (10.0-j)/10.0 * x1[limbs[i,0]-1] + j/10.0 * x1[limbs[i,1]-1]
			yij = (10.0-j)/10.0 * y1[limbs[i,0]-1] + j/10.0 * y1[limbs[i,1]-1]

			x1 = np.append(x1,xij)
			y1 = np.append(y1,yij)
			
			xij = (10.0-j)/10.0 * x2[limbs[i,0]-1] + j/10.0 * x2[limbs[i,1]-1]
			yij = (10.0-j)/10.0 * y2[limbs[i,0]-1] + j/10.0 * y2[limbs[i,1]-1]

			x2 = np.append(x2,xij)
			y2 = np.append(y2,yij)

	#v = makeFlowField(joints0,joints1,crop_size_x,crop_size_y,5)

	#I0 = cv2.GaussianBlur(I0, (3,3),0)
	#I1 = cv2.GaussianBlur(I1, (3,3),0)
	#I0 = cv2.resize(I0,(0,0),fx=dn_fac,fy=dn_fac)
	#I1 = cv2.resize(I1,(0,0),fx=dn_fac,fy=dn_fac)

	#H0_dn = np.zeros((int(H0.shape[0]*dn_fac), int(H0.shape[1]*dn_fac), n_joints))
	#H1_dn = np.zeros((int(H1.shape[0]*dn_fac), int(H1.shape[1]*dn_fac), n_joints))

	#for chan in xrange(n_joints):
	#	H0_dn[:,:,chan] = cv2.resize(H0[:,:,chan], (0,0), fx=dn_fac, fy=dn_fac)
	#	H1_dn[:,:,chan] = cv2.resize(H1[:,:,chan], (0,0), fx=dn_fac, fy=dn_fac)
'''



'''
def makeLimbHeatmaps(joints,crop_size_x,crop_size_y,stride):

	limbs = np.array([[0,1],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13], [2,5,8,11]])	
	#sigma_ys = np.array([11,11,11,11,11,11,11,11,11])	 

	H = np.zeros((crop_size_y/stride,crop_size_x/stride,limbs.shape[0]))

	x = np.array(range(H.shape[1]))*stride
	y = np.array(range(H.shape[0]))*stride
	xv,yv = np.meshgrid(x,y,sparse=False,indexing='xy')


	for i in xrange(limbs.shape[0]):		
		if(i < limbs.shape[0]-1):
			l1 = limbs[i][0]
			l2 = limbs[i][1]
			cx = (joints[l1,0] + joints[l2,0])/(2.0)
			cy = (joints[l1,1] + joints[l2,1])/(2.0)
			sigma_x = np.sqrt((joints[l1,0] - joints[l2,0])**2 + (joints[l1,1] - joints[l2,1])**2)/2
			sigma_y = 15
			theta = np.arctan2(joints[l2,1] - joints[l1,1], joints[l1,0] - joints[l2,0])
		else:
			cx = (joints[2,0] + joints[5,0] + joints[8,0] + joints[11,0])/4.0
			cy = (joints[2,1] + joints[5,1] + joints[8,1] + joints[11,1])/4.0
			sigma_x = 15			
			sigma_y = 15
			theta = 0

		a = np.cos(theta)**2/(2*sigma_x**2) + np.sin(theta)**2/(2*sigma_y**2)
		b = -np.sin(2*theta)/(4*sigma_x**2) + np.sin(2*theta)/(4*sigma_y**2)
		c = np.sin(theta)**2/(2*sigma_x**2) + np.cos(theta)**2/(2*sigma_y**2)

		H[:,:,i]  = np.exp(-(a*(xv-cx)*(xv-cx) + 2*b*(xv-cx)*(yv-cy) + c*(yv-cy)*(yv-cy)))

	return H 


def makeWarpedImageStack(I,j1,j2,crop_size_x,crop_size_y,stride):

	limbs = np.array([[0,1],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13], [2,5,8,11]])	

	Istack = np.zeros((crop_size_y/stride,crop_size_x/stride,3*limbs.shape[0]))

	x = np.array(range(Istack.shape[1]))*stride
	y = np.array(range(Istack.shape[0]))*stride
	xv,yv = np.meshgrid(x,y,sparse=False,indexing='xy')

	for i in xrange(limbs.shape[0]):	

		if(i < limbs.shape[0]-1):	
			l1 = limbs[i][0]
			l2 = limbs[i][1]
			p1 = np.array([[j1[l1,0],j1[l1,1]],[j1[l2,0],j1[l2,1]]])
			p2 = np.array([[j2[l1,0],j2[l1,1]],[j2[l2,0],j2[l2,1]]])
		else:
			p1 = np.array([[j1[2,0],j1[2,1]],[j1[5,0],j1[5,1]],[j1[8,0],j1[8,1]],[j1[11,0],j1[11,1]]])
			p2 = np.array([[j2[2,0],j2[2,1]],[j2[5,0],j2[5,1]],[j2[8,0],j2[8,1]],[j2[11,0],j2[11,1]]])

		tform,_ = transformations.make_similarity(p1,p2)
		M = np.array([[tform[1],-tform[3],tform[0]],[tform[3],tform[1],tform[2]]])
		Iw = cv2.warpAffine(I,M,(crop_size_x,crop_size_y))
		Istack[:,:,i*3:i*3+3] = cv2.resize(Iw,(0,0),fx=(1.0/stride), fy=(1.0/stride))
		
	return Istack


def makeInputOutputPair_spattransf(example,param):
	I0 = cv2.imread(example[0]) #,cv2.COLOR_BGR2HSV)
	I1 = cv2.imread(example[1]) #,cv2.COLOR_BGR2HSV)

	joints0 = np.reshape(np.array(example[2:30]), (14,2))
	joints1 = np.reshape(np.array(example[30:58]), (14,2))
	scale0 = example[61]/200.0
	scale1 = example[65]/200.0
	scale = np.max([scale0,scale1])
	pos0 = np.array([example[58] + example[60]/2.0, example[59] + example[61]/2.0])

	rshift = randShift(10)
	rscale = randScale(param)

	I0,joints0,pos0 = augScale(I0,scale,param,rscale,joints0,pos0)
	I1,joints1,_ = augScale(I1,scale,param,rscale,joints1)
	I0,joints0 = augCrop(I0,param,rshift,joints0,pos0)	
	I1,joints1 = augCrop(I1,param,rshift,joints1,pos0)	

	#I0 = (I0-128.0)/256.0	
	#I1 = (I1-128.0)/256.0

	crop_size_x = param['crop_size_x']
	crop_size_y = param['crop_size_y']
	stride = param['stride']
	sigma = param['sigma']

	H0 = np.zeros((crop_size_y/stride, crop_size_x/stride,n_joints))
	putGaussianMaps(H0,joints0,sigma,crop_size_x,crop_size_y,stride)
	
	H1 = np.zeros((crop_size_y/stride, crop_size_x/stride,n_joints))
	putGaussianMaps(H1,joints1,sigma,crop_size_x,crop_size_y,stride)

	Hlimb = makeLimbHeatmaps(joints1,crop_size_x,crop_size_y,stride)
	Ilimb = makeWarpedImageStack(I0,joints0,joints1,crop_size_x,crop_size_y,stride)

	return I0,I1,H0,H1,Hlimb,Ilimb
'''
