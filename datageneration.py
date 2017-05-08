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

def readImage(path):
	I = cv2.imread(path)
	I = (I/255.0 - 0.5)*2.0
	return I

def warpExampleGenerator(examples,param):
    
	img_width = param['IMG_WIDTH']
	img_height = param['IMG_HEIGHT']
	pose_dn = param['posemap_downsample']
	sigma_joint = param['sigma_joint']
	n_joints = param['n_joints']
	scale_factor = param['obj_scale_factor']	
	batch_size = param['batch_size']
	seq_len = param['seq_len']

	limbs = [[0,1],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13],[2,5,8,11]]	

	while True:

		ex_info = {}	
		for i in xrange(batch_size):
			r = (randScale(param),randShift(param),randRot(param),randSat(param))
			ex_info[i] = [np.random.randint(0,len(examples)), [], r]		

		for t in xrange(seq_len-1):

			X_src = np.zeros((batch_size,img_height,img_width,3))
			X_mask = np.zeros((batch_size,img_height,img_width,11))
			X_pose = np.zeros((batch_size,img_height/pose_dn,img_width/pose_dn,n_joints*2))
			X_trans = np.zeros((batch_size,2,3,11))
			Y = np.zeros((batch_size,img_height,img_width,3))
	
			for i in xrange(batch_size):
				example = examples[ex_info[i][0]] 

				I0 = readImage(example[0+t*33])
				I1 = readImage(example[0+(t+1)*33])

				joints0 = np.reshape(np.array(example[(t*33+1):(t*33+29)]), (14,2))
				joints1 = np.reshape(np.array(example[((t+1)*33+1):((t+1)*33+29)]), (14,2))

				scale = scale_factor/example[32]
				
				if(t==0):
					ex_info[i][1] = np.array([example[t*33+29] + example[t*33+31]/2.0, 
									example[t*33+30] + example[t*33+32]/2.0])

				I0,joints0 = centerAndScaleImage(I0,img_width,img_height,ex_info[i][1],scale,joints0)
				I1,joints1 = centerAndScaleImage(I1,img_width,img_height,ex_info[i][1],scale,joints1)

				rscale,rshift,rdegree,rsat = ex_info[i][2]
				I0,joints0 = augment(I0,joints0,rscale,rshift,rdegree,rsat,img_height,img_width)	
				I1,joints1 = augment(I1,joints1,rscale,rshift,rdegree,rsat,img_height,img_width)	

				posemap0 = makeJointHeatmaps(img_height,img_width,joints0,sigma_joint,pose_dn)
				posemap1 = makeJointHeatmaps(img_height,img_width,joints1,sigma_joint,pose_dn)

				limb_masks = makeLimbMasks(joints0,img_width,img_height,limbs)	
		
				fg_sig = (np.ptp(joints0,axis=0)/2.0)**2
				bg_mask = 1.0 - makeGaussianMap(img_width,img_height,
			    			np.mean(joints0,axis=0),fg_sig[0],fg_sig[1],0.0)
			
				X_src[i,:,:,:] = I0
				X_pose[i,:,:,0:n_joints] = posemap0
				X_pose[i,:,:,n_joints:] = posemap1
				X_mask[i,:,:,0] = bg_mask
				X_mask[i,:,:,1:] = limb_masks 
				X_trans[i,:,:,0] = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0]])
				X_trans[i,:,:,1:] = getLimbTransforms(joints0,joints1,img_width,img_height,limbs)
				Y[i,:,:,:] = I1
	
				yield ([X_src,X_pose,X_mask,X_trans],Y)


def augment(I,joints,rscale,rshift,rdegree,rsat,img_height,img_width):
	I,joints = augScale(I,rscale,joints)
	I,joints = augShift(I,img_width,img_height,rshift,joints)
	I,joints = augRotate(I,img_width,img_height,rdegree,joints)
	I = augSaturation(I,rsat)

	return I,joints


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

def augScale(I,scale_rand, joints):
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
	sigma_perp = np.array([9,5,5,5,5,5,5,5,5,11])**2	 

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

def getLimbTransforms(joints1,joints2,img_width,img_height,limbs):
	
	n_limbs = len(limbs)
	n_joints = joints1.shape[0]

	#Istack = np.zeros((img_height,img_width,3*n_limbs))
	Ms = np.zeros((2,3,n_limbs))

	for i in xrange(n_limbs):	

		n_joints_for_limb = len(limbs[i])
		p1 = np.zeros((n_joints_for_limb,2))
		p2 = np.zeros((n_joints_for_limb,2))

		for j in xrange(n_joints_for_limb):
			p1[j,:] = [joints1[limbs[i][j],0],joints1[limbs[i][j],1]]
			p2[j,:] = [joints2[limbs[i][j],0],joints2[limbs[i][j],1]]			

		tform,_ = transformations.make_similarity(p2,p1)
		M = np.array([[tform[1],-tform[3],tform[0]],[tform[3],tform[1],tform[2]]])
		#Iw = cv2.warpAffine(I,M,(img_width,img_height))
		#Istack[:,:,i*3:i*3+3] = Iw
		Ms[:,:,i] = M

	return Ms
