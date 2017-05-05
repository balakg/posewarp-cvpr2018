import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D,Dense,Activation,Input,UpSampling2D
from keras.layers import concatenate,Flatten,Reshape,Lambda,AveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.optimizers import Adam
import keras

def myConv(x_in,nf,ks=5,strides=1,activation='relu',ki='he_normal',name=None):

	if(name is None):
		x_out = Conv2D(nf,kernel_size=ks, padding='same',
			kernel_initializer=ki,strides=strides,activation=activation)(x_in)
	else:
		x_out = Conv2D(nf,kernel_size=ks, padding='same',
			kernel_initializer=ki,strides=strides,name=name,activation=activation)(x_in)

	return x_out

def myDense(x_in,nf,activation='relu'):
	x_out = Dense(nf,activation=activation,kernel_initializer='he_normal')(x_in)
	return x_out

'''
def posePredictor(n_joints,IMG_HEIGHT,IMG_WIDTH,IMG_CHAN,stride):

	x_img = Input(shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHAN))
	x_ctr = Input(shape=(IMG_HEIGHT/stride,IMG_WIDTH/stride,1))

	x0 = myConv(x_img,64,strides=2) #128x128x64
	x1 = myConv(x0,128,strides=2) #64x64x128
	x2 = myConv(x1,128,strides=2) #32x32x128
	x3 = myConv(x2,128,strides=2) #16x16x128
	x4 = myConv(x3,128,strides=2) #8x8x128

	x = UpSampling2D()(x4) #16x16x128
	x = concatenate([x,x3])
	x = myConv(x,128) #16x16x128
	x = UpSampling2D()(x) #32x32x128
	x = concatenate([x,x2])
	x = myConv(x,128) 
	x = UpSampling2D()(x) #64x64x128
	x = concatenate([x,x1])
	x = myConv(x,128)
	x = myConv(x,n_joints,activation='linear',ki='zeros')

	x_ctr_rep = concatenate([x_ctr,x_ctr,x_ctr,x_ctr,x_ctr,x_ctr,x_ctr,
						 x_ctr,x_ctr,x_ctr,x_ctr,x_ctr,x_ctr,x_ctr])

	y = keras.layers.add([x,x_ctr_rep])

	model = Model(inputs=[x_img,x_ctr],outputs=y,name='model_pose')
	adam = Adam(lr=5e-5)
	model.compile(optimizer=adam,loss='mse')

	return model


def posePredictor2(param):
	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	
	x_img = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))

	x = myConv(x_img,64,strides=2) #128x128x64
	x = myConv(x,128,strides=2) #64x64x128
	x = myConv(x,128,strides=2) #32x32x128
	x = myConv(x,128,strides=2) #16x16x128
	x = myConv(x,128,strides=2) #8x8x128
	

	x = Flatten()(x)
	x = Dense(1024,activation='relu',name='dense1',kernel_initializer='he_normal')(x)
	x = Dense(1024,activation='relu',name='dense2',kernel_initializer='he_normal')(x)
	x = Dense(n_joints*2,activation='linear',name='dense3',kernel_initializer='zeros')(x)

	y = Lambda(lambda x: x + 255.0/2.0)(x)

	model = Model(inputs=x_img,outputs=y,name='model_pose2')
	adam = Adam(lr=5e-4)
	model.compile(optimizer=adam,loss='mse')

	return model
'''

def make_trainable(net,val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

def poseDiscriminatorNet(model_gen, model_pose,param,feat_loss_weight):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']
	
	x_img = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	x_pose = Input(shape=(IMG_HEIGHT/pose_dn, IMG_WIDTH/pose_dn, n_joints*2))

	make_trainable(model_pose, False)	

	y_gen = model_gen([x_img,x_pose])
	y_gen_pose_feat = model_pose(y_gen)	

	model_combined = Model(inputs=[x_img,x_pose], outputs=[y_gen,y_gen_pose_feat],name='model_comb')
	adam = Adam(lr=1e-4)
	model_combined.compile(optimizer=adam,loss=['mse','mse'],loss_weights=[1.0,feat_loss_weight])
	return model_combined

def discriminator_original(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	x_src = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	x_tgt = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	x_pose = Input(shape=(IMG_HEIGHT/pose_dn, IMG_WIDTH/pose_dn, 2*n_joints))

	x = concatenate([x_src,x_tgt])
	x = myConv(x,64,strides=2) #64x64x128
	x = concatenate([x,x_pose]) #64x64x156
	x = myConv(x,128) #64x64x128
	x = myConv(x,128,strides=2) #32x32x128
	x = myConv(x,128,strides=2) #16x16x128
	x = myConv(x,128,strides=2) #8x8x128
	x = Flatten()(x)
	x = myDense(x,256,activation='relu')
	x = myDense(x,256,activation='relu')
	y = myDense(x,2,activation='softmax')

	model = Model(inputs=[x_src,x_tgt,x_pose],outputs=y, name='discriminator')
	return model

	
def discriminator_dense(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	x_src = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	x_tgt = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	x_pose = Input(shape=(IMG_HEIGHT/pose_dn, IMG_WIDTH/pose_dn, 2*n_joints))
	x_mask = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))

	def slice(arg):
		import tensorflow as tf
		return 1.0 - tf.expand_dims(arg[:,:,:,0],3)

	mask = Lambda(slice)(x_mask)

	x0 = concatenate([x_src,x_tgt])
	x1 = myConv(x0,64,strides=2) #64x64x128
	x2 = concatenate([x1,x_pose]) #64x64x156
	x3 = myConv(x2,128) #64x64x128
	x4 = myConv(x3,128,strides=2) #32x32x128
	x5 = myConv(x4,128,strides=2) #16x16x128
	x6 = myConv(x5,256,ks=3) #16x16x128
	
	x = UpSampling2D()(x6) #32x32x128
	x = concatenate([x,x4])
	x = myConv(x,128) #32x32x128
	x = UpSampling2D()(x) #64x64x128
	x = concatenate([x,x3])
	x = myConv(x,128) #64x64x128
	x = UpSampling2D()(x) #128x128x128
	x = concatenate([x,x0])
	x = myConv(x,128) #128x128x128
	x = myConv(x,1, activation='linear',ki='zeros')
	x = keras.layers.add([x,mask],name='discmap')

	y = GlobalMaxPooling2D()(x) 

	model = Model(inputs=[x_src,x_tgt,x_pose,x_mask],outputs=y, name='discriminator_dense')
	return model
	

def gan(generator,discriminator,param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	x_stack0 = Input(shape=(IMG_HEIGHT,IMG_WIDTH,33))
	x_pose0 = Input(shape=(IMG_HEIGHT/pose_dn, IMG_WIDTH/pose_dn, n_joints*2))
	x_mask0 = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	

	make_trainable(discriminator, False)
	y_gen = generator([x_stack0,x_pose0,x_mask0])

	x_src = Lambda(lambda arg: arg[:,:,:,0:3])(x_stack0)	
	y_class = discriminator([x_src,y_gen,x_pose0,x_mask0])

	gan = Model(inputs=[x_stack0,x_pose0,x_mask0], outputs=[y_gen,y_class], name='gan')
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

	num_batch = tf.shape(im)[0]
	height = tf.shape(im)[1]
	width = tf.shape(im)[2]
	channels = tf.shape(im)[3]
	
	#x = tf.reshape(x,[-1])
	#y = tf.reshape(y,[-1])

	x = tf.cast(x, 'float32')
	y = tf.cast(y, 'float32')

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
	base = _repeat(tf.range(num_batch)*dim1, height*width)

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
	output = tf.reshape(output, tf.stack([-1,height,width,channels]))
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


def makeWarpedStack(args):	
	mask = args[0]
	src_in = args[1]
	trans_in = args[2]

	for i in xrange(11):
		mask_i = K.repeat_elements(tf.expand_dims(mask[:,:,:,i],3),3,3)	
		src_masked =  tf.multiply(mask_i,src_in)
		warp_i = affineWarp(src_masked,trans_in[:,:,:,i])
		if(i == 0):
			warps = warp_i
		else:
			warps = tf.concat([warps,warp_i],3)

	return warps

def network_warp(param,feat_net):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_in = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints*2))
	mask_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))
	
	pose_src = Lambda(lambda arg: arg[:,:,:,0:14],
		output_shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))(pose_in)	

	x = concatenate([src_in,mask_in])
	x = myConv(x,64,strides=2)
	x = concatenate([x,pose_src])
	x = myConv(x,128)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)
	x = myConv(x,128)
	x = UpSampling2D()(x) 
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = myConv(x,128)
	x = UpSampling2D()(x)

	mask_delta = myConv(x,11,activation='linear',ki='zeros')
	mask = keras.layers.add([mask_delta,mask_in])

	def normalizeMask(arg):
		z = tf.reduce_max(arg,1,keep_dims=True)
		z = tf.reduce_max(z,2,keep_dims=True)
		z = tf.tile(z,[1,tf.shape(arg)[1],tf.shape(arg)[2],1])
		z = tf.divide(arg,z)
		return tf.clip_by_value(z,0.0,1.0)

	mask = Lambda(normalizeMask,name='mask')(mask)

	warp_stack = Lambda(makeWarpedStack,output_shape=(IMG_HEIGHT,IMG_WIDTH,33),
					name='warped_stack')([mask,src_in,trans_in])

	x0 = myConv(warp_stack,64)
	x1 = myConv(x0,64,strides=2) #64x64x64
	x2 = concatenate([x1,pose_in]) #64x64x92
	x3 = myConv(x2,128) #64x64x128
	x4 = myConv(x3,256,strides=2) #32x32x256
	x5 = myConv(x4,256,ks=3,strides=2) #16x16x256
	x6 = myConv(x5,256,ks=3,strides=2) #8x8x256
	x7 = myConv(x6,256,ks=3) #8x8x256

	x = UpSampling2D()(x7) #16x16x256
	x = concatenate([x,x5])
	x = myConv(x,256,ks=3) #16x16x256
	x = UpSampling2D()(x) #32x32x256
	x = concatenate([x,x4])
	x = myConv(x,256) #32x32x256
	x = UpSampling2D()(x) #64x64x256
	x = concatenate([x,x3]) #64x64x384
	x = myConv(x,128) #64x64x128
	x = UpSampling2D()(x) #128x128x128
	x = concatenate([x,x0])
	x = myConv(x,64)
	y = myConv(x,3,activation='tanh')#128x128x3	

	outputs = [y]
	if(feat_net is not None):
		def vgg_preprocess(arg):
			z = 255.0 * (arg+1.0)/2.0
			r = z[:,:,:,0] - 103.939
			g = z[:,:,:,1] - 116.779
			b = z[:,:,:,2] - 123.68
			return tf.stack([r,g,b],axis=3)

		y_vgg = Lambda(vgg_preprocess)(y)
		y_feat = feat_net(y_vgg)
		outputs.append(y_feat)
	
	model = Model(inputs=[src_in,pose_in,mask_in,trans_in],outputs=outputs)
	return model

'''
def network_matching(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']
	
	x_img = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3),name='img_input')
	x_pose = Input(shape=(IMG_HEIGHT/pose_dn, IMG_WIDTH/pose_dn, n_joints*2),name='pose_input')
	x_flow = Input(shape=(IMG_HEIGHT, IMG_WIDTH,2),name='flow_input')
 	x_mask = Input(shape=(IMG_HEIGHT, IMG_WIDTH,1),name='mask_input')
	
	x = myConv(x_img,64,strides=2) #128x128x64
	x = myConv(x,64,strides=2) #64x64x64
	x = concatenate([x,x_pose])
	x = myConv(x,128) #64x64x128
	x = myConv(x,128,strides=2) #32x32x128
	x = myConv(x,128,strides=2) #16x16x128
	x = myConv(x,128,strides=2) #8x8x128

	x = UpSampling2D()(x) #16x16x128
	x = myConv(x,128) #16x16x128
	x = UpSampling2D()(x) #32x32x128
	x = myConv(x,128) #32x32x128
	x = UpSampling2D()(x) #64x64x128
	x = myConv(x,64) #64x64x64
	x = UpSampling2D()(x) #128x128x64
	x = myConv(x,64) #128x128x64
	x = UpSampling2D()(x) #256x256x64

	fx = myConv(x,1,activation='linear',ki='zeros')
	fy = myConv(x,1,activation='linear',ki='zeros')
	mask = myConv(x,1,activation='linear',ki='zeros')	
	
	fx = Lambda(lambda x: x[0][:,:,:,0] + x[1][:,:,:,0],name='flowx')([fx,x_flow])
	fy = Lambda(lambda x: x[0][:,:,:,0] + x[1][:,:,:,1],name='flowy')([fy,x_flow])	
	y_fg = Lambda(interpolate,output_shape=(IMG_HEIGHT,IMG_WIDTH,3))([x_img,fx,fy])

	mask = keras.layers.add([mask,x_mask])
	mask = Activation('sigmoid',name='mask')(mask)	
	mask = concatenate([mask,mask,mask])

	y_fg_masked = keras.layers.multiply([y_fg,mask])
	inv_mask = Lambda(lambda x: 1-x) (mask)
	y_bg_masked = keras.layers.multiply([x_img,inv_mask])

	y = keras.layers.add([y_fg_masked,y_bg_masked])

	model = Model(inputs=[x_img,x_pose,x_flow,x_mask],outputs=y)
	adam = Adam(lr=1e-4)
	model.compile(optimizer=adam,loss='mse')

	return model

def motionNet(n_joints,IMG_HEIGHT,IMG_WIDTH,IMG_CHAN,stride,batch_size):
	x_img0 = Input(shape=(IMG_HEIGHT,IMG_WIDTH,IMG_CHAN),name='img_input')
	x_pose0 = Input(shape=(IMG_HEIGHT/stride, IMG_WIDTH/stride, n_joints*2),name='pose_input')
 
	x_mot = myConv(x_pose0,64) #64x64x64
	x_mot = myConv(x_mot,64,strides=2) #32x32x64
	x_mot = myConv(x_mot,128,strides=2) #16x16x128
	x_mot = myConv(x_mot,192,strides=2) #8x8x192


	#img_i = x_img_pyr[4]

	img_i = x_img0
	x_feats = []
	for i in xrange(0,5,1):
		xi = myConv(img_i,32)
		xi = myConv(xi,32)
		xi = myConv(xi,32)

		kerni = Lambda(lambda x: x[:,:,:,i*32:(i+1)*32])(x_mot) #batch_sizex8x8x6

		for j in xrange(batch_size):
			xij = Lambda(lambda x: tf.expand_dims(x[j,:,:,:],0))(xi) #1xhxwx3
			kernij = Lambda(lambda x: tf.expand_dims(x[j,:,:,:],axis=3))(kerni)

			xij_feat = Lambda(lambda x: tf.nn.depthwise_conv2d(x[0],x[1],
					[1,1,1,1],'SAME'))([xij,kernij])
			xij_feat = UpSampling2D((2**i,2**i))(xij_feat)

			if(j == 0):
				xi_feat = xij_feat
			else:
				xi_feat = Lambda(lambda x: K.concatenate([x[0],x[1]],axis=0))([xi_feat,xij_feat])	

		x_feats.append(xi_feat)
		img_i = AveragePooling2D()(img_i)

	x_enc = concatenate([x_feats[0],x_feats[1],x_feats[2],x_feats[3],x_feats[4]])	
	x_enc = myConv(x_enc,128)
	x_enc = myConv(x_enc,64)
	x_enc = myConv(x_enc,32)
	y = myConv(x_enc,3,activation='linear')	
	
	model = Model(inputs=[x_img0,x_pose0],outputs=y)
	return model
'''
