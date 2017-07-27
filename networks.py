import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D,Dense,Activation,Input,UpSampling2D,Dropout
from keras.layers import concatenate,Flatten,Reshape,Lambda
from keras.layers import LeakyReLU,GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
import keras

def myConv(x_in,nf,ks=3,strides=1,activation='lrelu',ki='he_normal',name=None,dropout=False):

	x_out = Conv2D(nf,kernel_size=ks, padding='same',kernel_initializer=ki,strides=strides)(x_in)
	#x_out = BatchNormalization()(x_out)

	if(dropout):
		x_out = Dropout(0.5)(x_out)

	if(name):
		if(activation == 'lrelu'):
			x_out = LeakyReLU(0.2,name=name)(x_out)
		else:
			x_out = Activation(activation,name=name)(x_out)
	else:
		if(activation == 'lrelu'):
			x_out = LeakyReLU(0.2)(x_out)
		else:
			x_out = Activation(activation)(x_out)

	return x_out


def myDense(x_in,nf,activation='relu',ki='he_normal',dropout=False):

	if(dropout):
		x_in = Dropout(0.5)(x_in)

	x_out = Dense(nf,activation=activation,kernel_initializer=ki)(x_in)
	return x_out


def make_trainable(net,val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

def discriminator(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	x_src = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	x_src_pose = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn, n_joints))
	x_tgt_pose = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn, n_joints))

	x = myConv(x_src,64,ks=5,strides=2)
	x = concatenate([x,x_src_pose,x_tgt_pose])
	x = myConv(x,128,ks=5,strides=2)
	x = myConv(x,128,ks=5,strides=2)
	x = myConv(x,256,ks=5,strides=2)
	x = myConv(x,256,strides=2) #8
	x = myConv(x,256)

	x = Flatten()(x)

	x = myDense(x,10)
	y = myDense(x,2,activation='softmax')

	model = Model(inputs=[x_src,x_src_pose,x_tgt_pose],outputs=y, name='discriminator')
	return model

def gan(generator,discriminator,param,feat_net,feat_weights,disc_loss,lr):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_src = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints))
	pose_tgt = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints))
	mask_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))

	make_trainable(discriminator, False)
	y_gen = generator([src_in,pose_src,pose_tgt,mask_in,trans_in])
	y_class = discriminator([y_gen,pose_src,pose_tgt])

	#Need to put this loss function in here for now because it relies on the extra
	#variable "feat_weights"
	def vggLoss(y_true,y_pred):
		y_true_feat = feat_net(Lambda(vgg_preprocess)(y_true))
		y_pred_feat = feat_net(Lambda(vgg_preprocess)(y_pred))

		loss = []
		for j in xrange(12):
			std = feat_weights[str(j)][1]
			std = tf.expand_dims(tf.expand_dims(tf.expand_dims(std,0),0),0)
			weights = tf.tile(std, [tf.shape(std)[0],tf.shape(std)[1],tf.shape(std)[2],1])

			d = tf.subtract(y_true_feat[j],y_pred_feat[j])
			loss_j = tf.reduce_mean(tf.abs(tf.divide(d,std)))
		
			if(j == 0):
				loss = loss_j
			else:
				loss = tf.add(loss,loss_j)

		return loss/12.0

	gan = Model(inputs=[src_in,pose_src,pose_tgt,mask_in,trans_in],outputs=[y_gen,y_class], name='gan')
	gan.compile(optimizer=Adam(lr=lr),loss=[vggLoss, 'binary_crossentropy'], loss_weights=[1.0,disc_loss])

	return gan


def _repeat(x, n_repeats):
	rep = tf.transpose(
		tf.expand_dims(tf.ones(shape=tf.stack([n_repeats,])), 1),[1,0])
	rep = tf.cast(rep, dtype='int32')
	x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
	return tf.reshape(x,[-1])

def _meshgrid(height, width):
	x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
          tf.transpose(tf.expand_dims(tf.linspace(0.0, 
		  tf.cast(width,tf.float32)-1.0, width), 1), [1, 0]))
	y_t = tf.matmul(tf.expand_dims(tf.linspace(0.0, 
		  tf.cast(height,tf.float32)-1.0, height), 1),
          tf.ones(shape=tf.stack([1, width])))
	return x_t,y_t

def interpolate(inputs):

	im = inputs[0]
	x = inputs[1]
	y = inputs[2]	

	im = tf.pad(im,[[0,0],[1,1],[1,1],[0,0]],"CONSTANT")
	
	num_batch = tf.shape(im)[0]
	height = tf.shape(im)[1]
	width = tf.shape(im)[2]
	channels = tf.shape(im)[3]
	
	x = tf.cast(x, 'float32')+1
	y = tf.cast(y, 'float32')+1

	max_x = tf.cast(width - 1, 'int32')
	max_y = tf.cast(height - 1, 'int32')

	x0 = tf.cast(tf.floor(x), 'int32')
	x1 = x0 + 1
	y0 = tf.cast(tf.floor(y), 'int32')
	y1 = y0 + 1

	x0 = tf.clip_by_value(x0, 0, max_x)
	x1 = tf.clip_by_value(x1, 0, max_x)
	y0 = tf.clip_by_value(y0, 0, max_y)
	y1 = tf.clip_by_value(y1, 0, max_y)

	dim2 = width
	dim1 = width*height
	base = _repeat(tf.range(num_batch)*dim1, (height-2)*(width-2))

	base_y0 = base + y0*dim2
	base_y1 = base + y1*dim2

	idx_a = base_y0 + x0
	idx_b = base_y1 + x0
	idx_c = base_y0 + x1
	idx_d = base_y1 + x1

	# use indices to lookup pixels in the flat image and restore
	# channels dim
	im_flat = tf.reshape(im, tf.stack([-1, channels]))
	im_flat = tf.cast(im_flat, 'float32')

	Ia = tf.gather(im_flat, idx_a)
	Ib = tf.gather(im_flat, idx_b)
	Ic = tf.gather(im_flat, idx_c)
	Id = tf.gather(im_flat, idx_d)

	# and finally calculate interpolated values
	x0_f = tf.cast(x0, 'float32')
	x1_f = tf.cast(x1, 'float32')
	y0_f = tf.cast(y0, 'float32')
	y1_f = tf.cast(y1, 'float32')

	dx = x1_f - x	
	dy = y1_f - y

	wa = tf.expand_dims((dx * dy), 1)
	wb = tf.expand_dims((dx * (1-dy)), 1)
	wc = tf.expand_dims(((1-dx) * dy), 1)
	wd = tf.expand_dims(((1-dx) * (1-dy)), 1)

	output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
	output = tf.reshape(output, tf.stack([-1,height-2,width-2,channels]))
	return output

def affineWarp(im,theta):	

	num_batch = tf.shape(im)[0]
	height = tf.shape(im)[1]
	width = tf.shape(im)[2]
	channels = tf.shape(im)[3]

	x_t,y_t = _meshgrid(height,width)
	x_t_flat = tf.reshape(x_t,(1,-1))
	y_t_flat = tf.reshape(y_t,(1,-1))
	ones = tf.ones_like(x_t_flat)
	grid = tf.concat(axis=0, values=[x_t_flat,y_t_flat,ones])
	grid = tf.expand_dims(grid,0)
	grid = tf.reshape(grid,[-1])
	grid = tf.tile(grid,tf.stack([num_batch]))
	grid = tf.reshape(grid,tf.stack([num_batch,3,-1]))

	T_g = tf.matmul(theta,grid)
	x_s = tf.slice(T_g, [0,0,0], [-1,1,-1])
	y_s = tf.slice(T_g, [0,1,0], [-1,1,-1])
	x_s_flat = tf.reshape(x_s,[-1])
	y_s_flat = tf.reshape(y_s,[-1])

	return interpolate([im,x_s_flat,y_s_flat])	

'''
def makeFlipWeights(args):
	src = args[0]
	tgt = args[1]

	x = concatenate([src, tgt])
	x = myDense(x,10)
	x = myDense(x,10)
	x = myDense(x,4,activation='sigmoid')
	return x
'''

def makeWarpedStack(args):	
	mask = args[0] # 11
	src_in = args[1]
	trans_in = args[2] # 15
	#flip_weights = args[3] #batch_size, 4

	idx = 0
	for i in xrange(11):
		mask_i = K.repeat_elements(tf.expand_dims(mask[:,:,:,i],3),3,3)	
		src_masked =  tf.multiply(mask_i,src_in)

		if(i == 0):
			src_masked = tf.add(src_masked,tf.multiply(1-mask_i,tf.random_uniform(tf.shape(src_in),-0.2,0.2)))
			warps = src_masked	
			idx += 1
		else:
			warp_i = affineWarp(src_masked,trans_in[:,:,:,idx])
			idx += 1
			warps = tf.concat([warps,warp_i],3)
			
			'''
			if(i >= 2 and i <= 5):
				warp_i_flip = affineWarp(src_masked,trans_in[:,:,:,idx])
				idx += 1
				warps = tf.concat([warps,warp_i_flip],3)
				
				weight = tf.expand_dims(tf.expand_dims(tf.expand_dims(flip_weights[:,i-2],1),1),1)	
				warp_i = tf.multiply(warp_i,1-weight)
				warp_i_flip = tf.multiply(warp_i_flip,weight)
				warp_i = tf.add(warp_i, warp_i_flip)
			
			warps = tf.concat([warps,warp_i],3)
			'''
	return warps


def vgg_preprocess(arg):
	z = 255.0 * (arg+1.0)/2.0
	r = z[:,:,:,0] - 103.939
	g = z[:,:,:,1] - 116.779
	b = z[:,:,:,2] - 123.68
	return tf.stack([r,g,b],axis=3)

def unet(x_in,pose_in,nf_enc,nf_dec,do_dropout):

	x0 = myConv(x_in,nf_enc[0],ks=7) #256
	x1 = myConv(x0,nf_enc[1],strides=2)#128
	x1 = concatenate([x1,pose_in])
	x2 = myConv(x1,nf_enc[2])
	x3 = myConv(x2,nf_enc[3],strides=2)#64
	x3 = myConv(x3,nf_enc[4])
	x4 = myConv(x3,nf_enc[5],strides=2)#32
	x4 = myConv(x4,nf_enc[6])
	x5 = myConv(x4,nf_enc[7],strides=2)#16
	x5 = myConv(x5,nf_enc[8])
	x6 = myConv(x5,nf_enc[9],strides=2)#8
	x = myConv(x6,nf_enc[10])

	skips = [x5,x4,x3,x2,x0]
	skip_nf = [nf_enc[8],nf_enc[6],nf_enc[4],nf_enc[2],nf_enc[0]]

	for i in xrange(5):
		x = UpSampling2D()(x)
		
		nf = min(128,skip_nf[i])

		x_skip = myConv(skips[i],nf)
		x_skip = myConv(x_skip,nf)

		x = concatenate([x,x_skip])
		x = myConv(x,nf_dec[i]) 

	return x


def network_fgbg(param,feat_net=None, feat_weights=None,do_dropout=False,loss='vgg'):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_src = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints))
	pose_tgt = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints))
	src_mask_prior = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))
	#tgt_mask_prior = Input(shape=(IMG_HEIGHT,IMG_WIDTH,1))

	#1. FG/BG separation
	x = unet(src_in,pose_src,[64]*2+[128]*9,[128]*4+[64],do_dropout)
	src_mask_delta = myConv(x,11,activation='linear',ki='zeros')
	src_mask = keras.layers.add([src_mask_delta,src_mask_prior])
	src_mask = Activation('softmax',name='mask_src')(src_mask)

	#2. Separate into fg limbs and background
	warped_stack = Lambda(makeWarpedStack)([src_mask,src_in,trans_in])
	fg_stack = Lambda(lambda arg: arg[:,:,:,3:],output_shape=(256,256,30),name='fg_stack')(warped_stack)	
	bg_src = Lambda(lambda arg: arg[:,:,:,0:3],output_shape=(256,256,3),name='bg_src')(warped_stack)
	bg_src_mask = Lambda(lambda arg: tf.expand_dims(arg[:,:,:,0],3))(src_mask)

	#3. BG synthesis
	nf_dec = [256,256,128,128,64]
	x = unet(concatenate([bg_src,bg_src_mask]),pose_src,[64]*2+[128]*2+[256]*7,nf_dec,do_dropout)
	bg_tgt = myConv(x,3,activation='tanh',name='bg_tgt')

	#4. FG synthesis
	pose_both = concatenate([pose_src,pose_tgt])
	x = unet(fg_stack,pose_both,[64]*2+[128]*2+[256]*7,nf_dec,do_dropout)
	fg_tgt = myConv(x,3,activation='tanh',name='fg_tgt')

	fg_mask = myConv(x,1,activation='sigmoid',name='fg_mask_tgt')
	fg_mask = concatenate([fg_mask,fg_mask,fg_mask])
	bg_mask = Lambda(lambda arg: 1-arg)(fg_mask)	

	#5. Merge bg and fg
	#fg_mask_delta = myConv(x,1,activation='linear',ki='zeros')
	#fg_mask = keras.layers.add([fg_mask_delta,tgt_mask_prior])
	#fg_mask = Activation('sigmoid',name='fg_mask_tgt')(fg_mask)
	#fg_mask = concatenate([fg_mask,fg_mask,fg_mask])
	#bg_mask = Lambda(lambda arg: 1-arg)(fg_mask)
	fg_tgt = keras.layers.multiply([fg_tgt,fg_mask],name='fg_tgt_masked')
	bg_tgt = keras.layers.multiply([bg_tgt,bg_mask],name='bg_tgt_masked')
	y = keras.layers.add([fg_tgt,bg_tgt])

	def vggLoss(y_true,y_pred):
		y_true_feat = feat_net(Lambda(vgg_preprocess)(y_true))
		y_pred_feat = feat_net(Lambda(vgg_preprocess)(y_pred))

		loss = []
		n_layers = 12
		for j in xrange(n_layers):
			std = feat_weights[str(j)][1]
			std = tf.expand_dims(tf.expand_dims(tf.expand_dims(std,0),0),0)
			weights = tf.tile(std, [tf.shape(std)[0],tf.shape(std)[1],tf.shape(std)[2],1])

			d = tf.subtract(y_true_feat[j],y_pred_feat[j])
			loss_j = tf.reduce_mean(tf.abs(tf.divide(d,std)))
		
			if(j == 0):
				loss = loss_j
			else:
				loss = tf.add(loss,loss_j)

		return loss/(1.0*n_layers)


	model = Model(inputs=[src_in,pose_src,pose_tgt,src_mask_prior,trans_in],outputs=[y])

	if(loss == 'l1'):
		model.compile(optimizer=Adam(lr=1e-4),loss='mae')	
	else:
		model.compile(optimizer=Adam(lr=1e-4),loss=[vggLoss])

	return model



def motionNet(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_src = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints))
	pose_tgt = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints))
	tgt_mask_prior = Input(shape=(IMG_HEIGHT,IMG_WIDTH,1))

	x = myConv(src_in,64,ks=3,strides=2) #128
	x = concatenate([pose_src,pose_tgt])
	x = myConv(x,128,ks=3,strides=2) #64	
	x = myConv(x,128,ks=3,strides=2) #32
	x = myConv(x,256,ks=3,strides=2) #16
	x = myConv(x,256,ks=3,strides=2) #8
	x = myConv(x,256,ks=3,strides=2) #4

	x = Flatten()(x)
	x = myDense(x,512)
	x = myDense(x,512)
	x = myDense(x,256*256*2,activation='linear')
	x = Lambda(lambda arg: arg + 128.0)(x)	
	x = Reshape((256,256,2),name='flow')(x)

	def call_interpolate(arg):
		x_s_flat = tf.reshape(arg[0][:,:,:,0],[-1])
		y_s_flat = tf.reshape(arg[0][:,:,:,1],[-1])
		return interpolate([arg[1],x_s_flat,y_s_flat])	

	x = Lambda(call_interpolate)([x,src_in])	

	tgt_mask = concatenate([tgt_mask_prior,tgt_mask_prior,tgt_mask_prior])
	y = keras.layers.multiply([x,tgt_mask])
	

	model = Model(inputs=[src_in,pose_src,pose_tgt,tgt_mask_prior],outputs=[y])
	model.compile(optimizer=Adam(lr=1e-4),loss='mae')	
	return model


'''
def posePred(args):
	joints = args[0]
	trans_in = args[1]

	limbs = [[0,1],[2,3],[3,4],[5,6],[6,7],[8,9],[9,10],[11,12],[12,13],[2,5,8,11]]
	
	pred = []	
	for joint in xrange(14):
		joint_x = joints[:,joint,0]
		joint_y = joints[:,joint,1]				

		pred_x = []
		pred_y = []
		ctr = 0
		for i in xrange(len(limbs)):
			for j in xrange(len(limbs[i])):
				if(limbs[i][j] != joint):
					continue;

				a0 = trans_in[:,0,0,i]
				a1 = trans_in[:,0,1,i]
				a2 = trans_in[:,0,2,i]
				a3 = trans_in[:,1,0,i]
				a4 = trans_in[:,1,1,i]
				a5 = trans_in[:,1,2,i]	

				tmp_x = tf.add(tf.add(tf.multiply(a0,joint_x),tf.multiply(a1,joint_y)),a2)
				tmp_y = tf.add(tf.add(tf.multiply(a3,joint_x),tf.multiply(a4,joint_y)),a5)
				
				if(ctr == 0):
					pred_x = tmp_x
					pred_y = tmp_y
				else:
					pred_x = tf.add(pred_x,tmp_x)
					pred_y = tf.add(pred_y,tmp_y)
			
				ctr = ctr + 1

		pred_x = pred_x/(ctr*1.0)	
		pred_y = pred_y/(ctr*1.0)

		pred_x = tf.expand_dims(tf.expand_dims(pred_x,1),2)
		pred_y = tf.expand_dims(tf.expand_dims(pred_y,1),2)
		pred_joint = tf.concat([pred_x,pred_y],axis=2)

		if(joint == 0):
			pred = pred_joint
		else:
			pred = tf.concat([pred,pred_joint],axis=1)

	return pred

def applyMask(args):
		bg = args[0]
		fg_stack = args[1]
		mask = args[2]

		res = []
		for i in xrange(11):
			mask_i = K.repeat_elements(tf.expand_dims(mask[:,:,:,i],3),3,3)	
		
			if(i == 0):
				res = tf.multiply(mask_i, bg)		
			else:
				res = tf.add(res,tf.multiply(mask_i, fg_stack[:,:,:,(i-1)*3:(i-1)*3+3])) 

		return res

	z = concatenate([Flatten()(posevec_src),Flatten()(posevec_tgt)])
	z = myDense(z,32)
	z = myDense(z,32)
	flip_prob = myDense(z,10,activation='sigmoid',ki='zeros')
	z = myDense(z,6*10,activation='linear',ki='zeros')

def gradient(arg):
	dy = arg[:,0:255,:,:] - arg[:,1:256,:,:]
	dx = arg[:,:,0:255,:] - arg[:,:,1:256,:]	

	#dy = tf.reduce_max(dy,axis=3,keep_dims=True)
	#dx = tf.reduce_max(dx,axis=3,keep_dims=True)

	batch_size = tf.shape(arg)[0]
	height = tf.shape(arg)[1]
	width = tf.shape(arg)[2]
	n_chan = tf.shape(arg)[3]

	dy = tf.concat([dy,tf.zeros([batch_size,1,width,n_chan],tf.float32)],axis=1)
	dx = tf.concat([dx,tf.zeros([batch_size,height,1,n_chan],tf.float32)],axis=2)	

	return tf.concat([dx,dy],axis=3)

'''
