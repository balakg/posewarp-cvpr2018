import numpy as np
import json
import cv2
import scipy.io as sio
from scipy import interpolate
import transformations

def randScale(param):
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

'''
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


def makeInitialWarpField(joints0,joints1,sigma,img_width,img_height):
	limbs = np.array([[1,2],[3,4],[4,5],[6,7],[7,8],[9,10],[10,11],[12,13],[13,14],[3,9],[6,12],[3,6],[9,12]])	

	joints0 = augmentJoints(joints0,20,limbs)
	joints1 = augmentJoints(joints1,20,limbs)

	xv,yv = np.meshgrid(np.array(range(img_width)),np.array(range(img_height)),sparse=False,indexing='xy')

	V = np.zeros((img_height,img_width,2))
	M = np.zeros((img_height,img_width))

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

	M = np.reshape(M,(img_height,img_width,1))

	return V
'''

def transferExampleGenerator(examples,batch_size,param):
    
	img_width = param['IMG_WIDTH']
	img_height = param['IMG_HEIGHT']
	pose_dn = param['posemap_downsample']
	sigma_joint = param['sigma_joint']
	n_joints = param['n_joints']
	scale_factor = param['obj_scale_factor']	

	X_src = np.zeros((batch_size,img_height,img_width,11*3)) #source image + 10 warped ones
	X_tgt = np.zeros((batch_size,img_height,img_width,3))
	X_mask = np.zeros((batch_size,img_height,img_width,11))
	X_pose = np.zeros((batch_size,img_height/pose_dn,img_width/pose_dn,n_joints*2))

	#limbs: head, right upper arm, right lower arm, left upper arm, left lower arm,
	#right upper leg, right lower leg, left upper leg, left lower leg, chest
	limbs = [[0,1],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13],[2,5,8,11]]	

	while True:
		for i in xrange(batch_size):
			example = examples[np.random.randint(0,len(examples))] 
			I0 = cv2.imread(example[0])
			I1 = cv2.imread(example[1])

			I0 = I0/255.0 - 0.5
			I1 = I1/255.0 - 0.5

			joints0 = np.reshape(np.array(example[2:30]), (14,2))
			joints1 = np.reshape(np.array(example[30:58]), (14,2))
			scale = scale_factor/np.max([example[61],example[65]])
			
			pos0 = np.array([example[58] + example[60]/2.0, example[59] + example[61]/2.0])
			pos1 = np.array([example[62] + example[64]/2.0, example[63] + example[65]/2.0])

			#Center and scale images. Center second image using position of first image so that
			#the whole scene doesn't translate.
			I0,joints0 = centerAndScaleImage(I0,img_width,img_height,pos0,scale,joints0)
			I1,joints1 = centerAndScaleImage(I1,img_width,img_height,pos0,scale,joints1) 

			#Data augmentation.			
			rscale = randScale(param)
			rshift = randShift(param)
			rdegree = randRot(param)
			rsat = randSat(param)

			I0,joints0 = augScale(I0,scale,rscale,joints0)
			I1,joints1 = augScale(I1,scale,rscale,joints1)	

			I0,joints0 = augShift(I0,img_width,img_height,rshift,joints0)
			I1,joints1 = augShift(I1,img_width,img_height,rshift,joints1)	

			I0,joints0 = augRotate(I0,img_width,img_height,rdegree,joints0)
			I1,joints1 = augRotate(I1,img_width,img_height,rdegree,joints1)

			I0 = augSaturation(I0,rsat)
			I1 = augSaturation(I1,rsat)

			#Construct heatmaps for joints
			posemap0 = makeJointHeatmaps(img_height,img_width,joints0,sigma_joint,pose_dn)
			posemap1 = makeJointHeatmaps(img_height,img_width,joints1,sigma_joint,pose_dn)

			#Warp the source image once for each limb
			I0_warps = makeWarpedImageStack(I0,joints0,joints1,img_width,img_height,limbs)

			#Make gaussian masks for the limbs in the warped images. Also make a background mask,
			#to be applied to the original image in the neural network.
			limb_masks = makeLimbMasks(joints1,img_width,img_height,limbs)	
			fg_sigmas = (np.ptp(joints1,axis=0)/2.0)**2

			bg_mask = 1.0 - makeGaussianMap(img_width,img_height,np.mean(joints1,axis=0),fg_sigmas[0],fg_sigmas[1],0.0)

			X_src[i,:,:,0:3] = I0
			X_src[i,:,:,3:] = I0_warps

			X_mask[i,:,:,0] = bg_mask
			X_mask[i,:,:,1:] = limb_masks 

			X_tgt[i,:,:,:] = I1
			X_pose[i,:,:,:] = np.concatenate((posemap0,posemap1),axis=2)	
	
		yield (X_src,X_tgt,X_pose,X_mask)

'''
def poseExampleGenerator(examples,batch_size,param):
    
	img_width = param['IMG_WIDTH']
	img_height = param['IMG_HEIGHT']
	stride = param['posemap_downsample']
	sigma = param['sigma']
	n_joints = param['n_joints']
	
	X_img = np.zeros((batch_size,img_height,img_width,3))
	#X_pose = np.zeros((batch_size,img_height/stride,img_width/stride,n_joints*2))
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

			#H0 = np.zeros((img_height/stride, img_width/stride,n_joints))
			#putGaussianMaps(H0,joints0,sigma,stride)

			X_img[i,:,:,:] = I
			X_pose[i,:] = joints.flatten()

		yield (X_img,X_pose)
'''

def centerAndScaleImage(I,img_width,img_height,pos,scale,joints):

	I = cv2.resize(I,(0,0),fx=scale,fy=scale)
	joints = joints * scale

	x_offset = (img_width-1.0)/2.0 - pos[0]*scale
	y_offset = (img_height-1.0)/2.0 - pos[1]*scale

	T = np.float32([[1,0,x_offset],[0,1,y_offset]])	
	I = cv2.warpAffine(I,T,(img_width,img_height))

	joints[:,0] += x_offset
	joints[:,1] += y_offset

	return I,joints

def augScale(I,obj_scale,scale_rand, joints):
	I = cv2.resize(I,(0,0),fx=scale_rand,fy=scale_rand)
	joints = joints * scale_rand
	return I,joints


def augRotate(I,img_width,img_height,degree_rand,joints):
	h = I.shape[0]
	w = I.shape[1]	

	center = ( (w-1.0)/2.0, (h-1.0)/2.0 )
	R = cv2.getRotationMatrix2D(center,degree_rand,1)	
	I = cv2.warpAffine(I,R,(img_width,img_height))

	for i in xrange(joints.shape[0]):
		joints[i,:] = rotatePoint(joints[i,:],R)

	return I,joints

def rotatePoint(p,R):
	x_new = R[0,0] * p[0] + R[0,1] * p[1] + R[0,2]
	y_new = R[1,0] * p[0] + R[1,1] * p[1] + R[1,2]	
	return np.array((x_new,y_new))


def augShift(I,img_width,img_height,rand_shift,joints):
	x_shift = rand_shift[0]
	y_shift = rand_shift[1]

	T = np.float32([[1,0,x_shift],[0,1,y_shift]])	
	I = cv2.warpAffine(I,T,(img_width,img_height))

	joints[:,0] += x_shift
	joints[:,1] += y_shift

	return I,joints

def augSaturation(I,rsat):
	I  *= rsat
	I[I > 1] = 1
	return I

def makeJointHeatmaps(height,width,joints,sigma,pose_dn):

	height = height/pose_dn
	width = width/pose_dn
	sigma = (sigma/pose_dn)**2
	joints = joints/pose_dn

	H = np.zeros((height,width,joints.shape[0]))

	for i in xrange(H.shape[2]):	
		H[:,:,i] = makeGaussianMap(width,height,joints[i,:],sigma,sigma,0.0)

	return H

def makeGaussianMap(img_width,img_height,center,sigma_x,sigma_y,theta):

	xv,yv = np.meshgrid(np.array(range(img_width)),np.array(range(img_height)),
						sparse=False,indexing='xy')
	
	a = np.cos(theta)**2/(2*sigma_x) + np.sin(theta)**2/(2*sigma_y)
	b = -np.sin(2*theta)/(4*sigma_x) + np.sin(2*theta)/(4*sigma_y)
	c = np.sin(theta)**2/(2*sigma_x) + np.cos(theta)**2/(2*sigma_y)

	return np.exp(-(a*(xv-center[0])*(xv-center[0]) + 
			2*b*(xv-center[0])*(yv-center[1]) + 
			c*(yv-center[1])*(yv-center[1])))


def makeLimbMasks(joints,img_width,img_height,limbs):
	n_limbs = len(limbs)
	n_joints = joints.shape[0]

	mask = np.zeros((img_height,img_width,n_limbs))

	#Gaussian sigma perpendicular to the limb axis. I hardcoded
	#reasonable sigmas for now.
	sigma_perp = np.array([11,5,5,5,5,5,5,5,5,11])**2	 

	for i in xrange(n_limbs):	
		n_joints_for_limb = len(limbs[i])
		p = np.zeros((n_joints_for_limb,2))

		for j in xrange(n_joints_for_limb):
			p[j,:] = [joints[limbs[i][j],0],joints[limbs[i][j],1]]

		if(n_joints_for_limb == 4):
			p_top = np.mean(p[0:2,:],axis=0)
			p_bot = np.mean(p[2:4,:],axis=0)
			p = np.vstack((p_top,p_bot))

		center = np.mean(p,axis=0)		

		sigma_parallel = (np.sum((p[1,:] - p[0,:])**2))/4
		theta = np.arctan2(p[1,1] - p[0,1], p[0,0] - p[1,0])

		mask_i = makeGaussianMap(img_width,img_height,center,sigma_parallel,sigma_perp[i],theta)
		mask[:,:,i] = mask_i/np.amax(mask_i)
		
	return mask

def makeWarpedImageStack(I,joints1,joints2,img_width,img_height,limbs):
	
	n_limbs = len(limbs)
	n_joints = joints1.shape[0]

	Istack = np.zeros((img_height,img_width,3*n_limbs))

	for i in xrange(n_limbs):	

		n_joints_for_limb = len(limbs[i])
		p1 = np.zeros((n_joints_for_limb,2))
		p2 = np.zeros((n_joints_for_limb,2))

		for j in xrange(n_joints_for_limb):
			p1[j,:] = [joints1[limbs[i][j],0],joints1[limbs[i][j],1]]
			p2[j,:] = [joints2[limbs[i][j],0],joints2[limbs[i][j],1]]			

		tform,_ = transformations.make_similarity(p1,p2)
		M = np.array([[tform[1],-tform[3],tform[0]],[tform[3],tform[1],tform[2]]])
		Iw = cv2.warpAffine(I,M,(img_width,img_height))
		Istack[:,:,i*3:i*3+3] = Iw
		
	return Istack
