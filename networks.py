import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D,Dense,Activation,Input,UpSampling2D
from keras.layers import concatenate,Flatten,Reshape,Lambda
from keras.layers import SimpleRNN,TimeDistributed,LeakyReLU
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
import keras


def myConv(x_in,nf,ks=5,strides=1,activation='relu',ki='he_normal',name=None,td=False):

	if(name is None):
		layer = Conv2D(nf,kernel_size=ks, padding='same',
			kernel_initializer=ki,strides=strides,activation=activation)
	else:
		layer = Conv2D(nf,kernel_size=ks, padding='same',
			kernel_initializer=ki,strides=strides,name=name,activation=activation)

	if(td):
		x_out = TimeDistributed(layer)(x_in)
	else:
		x_out = layer(x_in)

	return x_out

def myDense(x_in,nf,activation='relu',ki='he_normal'):
	x_out = Dense(nf,activation=activation,kernel_initializer=ki)(x_in)
	return x_out


def make_trainable(net,val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

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

def discriminator(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	x_src = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	x_pose = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn, n_joints))
	#x_mask = Input(shape=(IMG_HEIGHT,IMG_WIDTH,1))

	#x_grad = Lambda(gradient)(x_src)
	
	#x = concatenate([x_mask,x_mask,x_mask,x_mask,x_mask,x_mask])
	#x = keras.layers.multiply([x,x_grad])
	x = myConv(x_src,32,strides=2)
	x = concatenate([x,x_pose])
	x = myConv(x,64,strides=2)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)
	x = myConv(x,256,strides=2)

	x = Flatten()(x)
	x = myDense(x,10,activation='relu')
	y = myDense(x,2,activation='softmax')

	model = Model(inputs=[x_src,x_pose],outputs=y, name='discriminator')
	return model

def gan(generator,discriminator,param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_in = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints*2))
	mask_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))
	#mask_tgt = Input(shape=(IMG_HEIGHT,IMG_WIDTH,1))

	make_trainable(discriminator, False)
	y_gen = generator([src_in,pose_in,mask_in,trans_in])

	pose_tgt = Lambda(lambda arg: arg[:,:,:,14:],
		output_shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))(pose_in)	

	y_class = discriminator([y_gen[0],pose_tgt])

	gan = Model(inputs=[src_in,pose_in,mask_in,trans_in], 
				outputs=[y_gen[0],y_class], name='gan')
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

def vgg_preprocess(arg):
	z = 255.0 * (arg+1.0)/2.0
	r = z[:,:,:,0] - 103.939
	g = z[:,:,:,1] - 116.779
	b = z[:,:,:,2] - 123.68
	return tf.stack([r,g,b],axis=3)


def gradientLoss(y_true,y_pred):
	dx_true = y_true[:,0:255,:,:] - y_true[:,1:256,:,:]
	dy_true = y_true[:,:,0:255,:] - y_true[:,:,1:256,:]	
	dx_pred = y_pred[:,0:255,:,:] - y_pred[:,1:256,:,:]
	dy_pred = y_pred[:,:,0:255,:] - y_pred[:,:,1:256,:]	
	dx2 = tf.reduce_mean(tf.multiply(dx_true-dx_pred,dx_true-dx_pred))		
	dy2 = tf.reduce_mean(tf.multiply(dy_true-dy_pred,dy_true-dy_pred))			
	energy = dx2 + dy2
	return energy

'''
def network_warp(param,feat_net=None):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_in = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints*2))
	mask_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))
	
	pose_src = Lambda(lambda arg: arg[:,:,:,0:n_joints])(pose_in)	
	
	x = myConv(src_in,64,strides=2)
	x = concatenate([x,pose_src])
	x = myConv(x,128)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)
	x = myConv(x,128)
	x = UpSampling2D()(x) 
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = myConv(x,128)
	x = UpSampling2D()(x)

	mask_delta = myConv(x,11,activation='linear',ki='zeros')
	mask = keras.layers.add([mask_delta,mask_in])
	mask = Activation('softmax',name='mask')(mask)

	warped_stack = Lambda(makeWarpedStack,output_shape=(IMG_HEIGHT,IMG_WIDTH,33),
					name='warped_stack')([mask,src_in,trans_in])
	
	x0 = myConv(warped_stack,128,strides=2)
	x1 = concatenate([x0,pose_in])
	x2 = myConv(x1,128)
	x3 = myConv(x2,128,strides=2) #64
	x4 = myConv(x3,256,strides=2) #32x32x256
	x5 = myConv(x4,256,strides=2) #16x16x256
	x6 = myConv(x5,256,ks=3,strides=2) #8x8x256
	x7 = myConv(x6,256,ks=3) #8x8x256

	x = UpSampling2D()(x7) #16x16x256
	x = concatenate([x,x5])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #32x32x256
	x = concatenate([x,x4])
	x = myConv(x,256)
	x = UpSampling2D()(x) #64x64x256
	x = concatenate([x,x3])
	x = myConv(x,128)
	x = UpSampling2D()(x) #128x128x128
	x = concatenate([x,x2])
	x = UpSampling2D()(x) #256x256
	x = myConv(x,64)
	y = myConv(x,3,activation='tanh')#256x256x3	

	outputs = [y]
	if(feat_net is not None):		
		y = Lambda(vgg_preprocess)(y)
		y_feat = feat_net(y)
		outputs.append(y_feat)

	model = Model(inputs=[src_in,pose_in,mask_in,trans_in],outputs=outputs)
	#model.compile(optimizer=Adam(lr=1e-4),loss=['mse',gradientLoss,'mse'],loss_weights=[1.0,1.0,0.001])
	return model
'''

def network_fgbg(param,feat_net=None):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_in = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints*2))
	mask_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))
	
	pose_src = Lambda(lambda arg: arg[:,:,:,0:n_joints])(pose_in)	
	
	x = myConv(src_in,128,strides=2)
	x = concatenate([x,pose_src])
	x = myConv(x,128)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)
	x = UpSampling2D()(x) 
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = myConv(x,128)

	mask_delta = myConv(x,11,activation='linear',ki='zeros')
	mask = keras.layers.add([mask_delta,mask_in])
	mask = Activation('softmax',name='mask_src')(mask)

	warped_stack = Lambda(makeWarpedStack)([mask,src_in,trans_in])

	fg_stack = Lambda(lambda arg: arg[:,:,:,3:],output_shape=(256,256,30),
				name = 'fg_stack')(warped_stack)	
	bg_src = Lambda(lambda arg: arg[:,:,:,0:3],output_shape=(256,256,3),
				name = 'bg_src')(warped_stack)
	bg_src_mask = Lambda(lambda arg: tf.expand_dims(arg[:,:,:,0],3))(mask)

	bg_src = concatenate([bg_src,bg_src_mask])
	x0 = myConv(bg_src,128,strides=2)
	x1 = myConv(x0,128,strides=2) #64
	x2 = myConv(x1,256,strides=2) #32x32x256
	x3 = myConv(x2,256,strides=2) #16x16x256
	x4 = myConv(x3,256,ks=3,strides=2) #8x8x256

	x = UpSampling2D()(x4) #16x16x256
	x = concatenate([x,x3])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #32x32x256
	x = concatenate([x,x2])
	x = myConv(x,256)
	x = UpSampling2D()(x) #64x64x256
	x = concatenate([x,x1])
	x = myConv(x,128)
	x = UpSampling2D()(x) #128
	x = concatenate([x,x0])
	x = myConv(x,128)
	x = UpSampling2D()(x) #256
	x = concatenate([x,bg_src])
	x = myConv(x,32)
	bg_tgt = myConv(x,3,activation='tanh',name='bg_tgt')


	x0 = myConv(fg_stack,128,strides=2)
	x1 = concatenate([x0,pose_in])
	x2 = myConv(x1,128)
	x3 = myConv(x2,128,strides=2) #64
	x4 = myConv(x3,256,strides=2) #32x32x256
	x5 = myConv(x4,256,strides=2) #16x16x256
	x6 = myConv(x5,256,ks=3,strides=2) #8x8x256
	x7 = myConv(x6,256,ks=3) #8x8x256

	x = UpSampling2D()(x7) #16x16x256
	x = concatenate([x,x5])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #32x32x256
	x = concatenate([x,x4])
	x = myConv(x,256)
	x = UpSampling2D()(x) #64x64x256
	x = concatenate([x,x3])
	x = myConv(x,128)
	x = UpSampling2D()(x) #128x128x128
	x = concatenate([x,x2])
	x = UpSampling2D()(x) #256x256
	fg_tgt = myConv(x,3,activation='tanh',name='fg_tgt')#256x256x3
	fg_mask = myConv(x,1,activation='sigmoid',name='fg_mask_tgt')

	fg_mask = concatenate([fg_mask,fg_mask,fg_mask])
	bg_mask = Lambda(lambda arg: 1-arg)(fg_mask)	

	fg_tgt = keras.layers.multiply([fg_tgt,fg_mask])
	bg_tgt = keras.layers.multiply([bg_tgt,bg_mask])
	y = keras.layers.add([fg_tgt,bg_tgt])

	outputs = [y]
	if(feat_net is not None):		
		y = Lambda(vgg_preprocess)(y)
		y_feat = feat_net(y)
		outputs.append(y_feat)

	model = Model(inputs=[src_in,pose_in,mask_in,trans_in],outputs=outputs)
	return model



def network_ed(param,feat_net=None):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_in = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints*2))
	
	x0 = myConv(src_in,128,strides=2)
	x1 = concatenate([x0,pose_in])
	x2 = myConv(x1,128)
	x3 = myConv(x2,128,strides=2) #64
	x4 = myConv(x3,256,strides=2) #32x32x256
	x5 = myConv(x4,256,strides=2) #16x16x256
	x6 = myConv(x5,256,ks=3,strides=2) #8x8x256
	x7 = myConv(x6,256,ks=3) #8x8x256

	x = UpSampling2D()(x7) #16x16x256
	x = concatenate([x,x5])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #32x32x256
	x = concatenate([x,x4])
	x = myConv(x,256)
	x = UpSampling2D()(x) #64x64x256
	x = concatenate([x,x3])
	x = myConv(x,128)
	x = UpSampling2D()(x) #128x128x128
	x = concatenate([x,x2])
	x = UpSampling2D()(x) #256x256
	x = myConv(x,64)
	y = myConv(x,3,activation='tanh')#256x256x3	

	outputs = [y]
	if(feat_net is not None):		
		y = Lambda(vgg_preprocess)(y)
		y_feat = feat_net(y)
		outputs.append(y_feat)

	model = Model(inputs=[src_in,pose_in],outputs=outputs)
	return model


def network_warpclass(param,feat_net=None):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_in = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,n_joints*2))
	mask_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))
	class_src = Input(shape=(3,))
	class_tgt = Input(shape=(3,))	

	pose_src = Lambda(lambda arg: arg[:,:,:,0:n_joints])(pose_in)	
	
	x = myConv(src_in,64,strides=2)
	x = concatenate([x,pose_src])
	x = myConv(x,128)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)
	x = myConv(x,128,strides=2)

	x = Flatten()(x)
	x = concatenate([x,class_src])
	x = myDense(x,512)
	x = myDense(x,512)

	x = myDense(x,8*8*128)
	x = Reshape((8,8,128))(x)

	x = UpSampling2D()(x)
	x = myConv(x,128)
	x = UpSampling2D()(x) 
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = myConv(x,128)
	x = UpSampling2D()(x)

	mask_delta = myConv(x,11,activation='linear',ki='zeros')
	mask = keras.layers.add([mask_delta,mask_in])
	mask = Lambda(normalizeMask,name='mask')(mask)

	warped_stack = Lambda(makeWarpedStack,output_shape=(IMG_HEIGHT,IMG_WIDTH,33),
					name='warped_stack')([mask,src_in,trans_in])
	
	x0 = myConv(warped_stack,128,strides=2)
	x1 = concatenate([x0,pose_in])
	x2 = myConv(x1,128)
	x3 = myConv(x2,128,strides=2) #64
	x4 = myConv(x3,256,strides=2) #32x32x256
	x5 = myConv(x4,256,strides=2) #16x16x256
	x6 = myConv(x5,256,ks=3,strides=2) #8x8x256
	x7 = myConv(x6,256,ks=3,strides=2) #4x4x256

	x = Flatten()(x7)
	x = concatenate([x,class_tgt])
	x = myDense(x,1024)
	x = myDense(x,1024)
	x = myDense(x,4*4*256)
	x = Reshape((4,4,256))(x)	

	x = UpSampling2D()(x) #8x8x256
	x = concatenate([x,x6])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #16x16x256
	x = concatenate([x,x5])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #32x32x256
	x = concatenate([x,x4])
	x = myConv(x,256)
	x = UpSampling2D()(x) #64x64x256
	x = concatenate([x,x3])
	x = myConv(x,128)
	x = UpSampling2D()(x) #128x128x128
	x = concatenate([x,x2])
	x = UpSampling2D()(x) #256x256
	x = myConv(x,64)
	y = myConv(x,3,activation='tanh')#256x256x3	

	outputs = [y]
	if(feat_net is not None):		
		y = Lambda(vgg_preprocess)(y)
		y_feat = feat_net(y)
		outputs.append(y_feat)

	model = Model(inputs=[src_in,pose_in,mask_in,trans_in,class_src,class_tgt],outputs=outputs)
	#model.compile(optimizer=Adam(lr=1e-4),loss=['mse',gradientLoss,'mse'],loss_weights=[1.0,1.0,0.001])
	return model

'''
def posenet(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	ctr_in = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,1))
	
	x0 = myConv(src_in,128,strides=2)#128
	x1 = myConv(x0,128,strides=2)#64
	x2 = myConv(x1,128,strides=2)#32
	x3 = myConv(x2,128,strides=2)#16
	x4 = myConv(x3,128,strides=2,ks=3)
	x5 = myConv(x4,128,ks=3)

	x = UpSampling2D()(x5) #16
	x = concatenate([x,x3])
	x = myConv(x,128)
	x = UpSampling2D()(x) #32
	x = concatenate([x,x2])
	x = myConv(x,128)
	x = UpSampling2D()(x) #64
	x = concatenate([x,x1])
	x = myConv(x,128)
	x = UpSampling2D()(x)
	x = concatenate([x,x0])
	y = myConv(x,n_joints,activation='linear',ki='zeros')

	ctr_rep = concatenate([ctr_in,ctr_in,ctr_in,ctr_in,ctr_in,ctr_in,ctr_in,
							 ctr_in,ctr_in,ctr_in,ctr_in,ctr_in,ctr_in,ctr_in])

	y = keras.layers.add([y,ctr_rep])
	
	model = Model(inputs=[src_in,ctr_in],outputs=y)
	return model


def network_warp_affine(param,feat_net=None):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,n_joints*2))
	mask_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))
	
	x0 = concatenate([src_in,pose_in])
	x1 = myConv(x0,64,strides=2) #64
	x2 = myConv(x1,128,strides=2) #32
	x3 = myConv(x2,128,strides=2) #16
	x4 = myConv(x3,128,strides=2,ks=3) #8
	x5 = myConv(x4,128,ks=3)#8
	x = UpSampling2D()(x5) #16x16x128
	x = myConv(x,128,ks=3) #16x16x128
	x = UpSampling2D()(x) #32x32x128
	x = myConv(x,128) #32x32x128
	x = UpSampling2D()(x) #64x64x128
	x = myConv(x,128) #64x64x128
	x = UpSampling2D()(x) #128x128x128

	mask_delta = myConv(x,11,activation='linear')
	mask = keras.layers.add([mask_delta,mask_in])
	mask = Lambda(normalizeMask,name='mask')(mask)

	trans = Flatten()(x5)
	trans = myDense(trans,128)
	trans = myDense(trans,6*11,activation='linear')
	trans = Reshape((2,3,11))(trans)
	trans = keras.layers.add([trans,trans_in],name='transforms')

	warped_stack = Lambda(makeWarpedStack,
					output_shape=(IMG_HEIGHT,IMG_WIDTH,3*11),
					name='warped_stack')([mask,src_in,trans])

	x0 = concatenate([warped_stack,pose_in])
	x1 = myConv(x0,128,strides=2)
	x2 = myConv(x1,256,strides=2) #32x32x256
	x3 = myConv(x2,256,strides=2) #16x16x256
	x4 = myConv(x3,256,ks=3,strides=2) #8x8x256
	x5 = myConv(x4,256,ks=3) #8x8x256

	x = UpSampling2D()(x5) #16x16x256
	x = concatenate([x,x3])
	x = myConv(x,256,ks=3) #16x16x256
	x = UpSampling2D()(x) #32x32x256
	x = concatenate([x,x2])
	x = myConv(x,256) #32x32x256
	x = UpSampling2D()(x) #64x64x256
	x = concatenate([x,x1]) #64x64x384
	x = myConv(x,128) #64x64x128
	x = UpSampling2D()(x) #128x128x128
	x = concatenate([x,x0])
	x = myConv(x,64)
	y = myConv(x,3,activation='tanh')#128x128x3	

	outputs = [y]
	if(feat_net is not None):		
		y = Lambda(vgg_preprocess)(y)
		y_feat = feat_net(y)
		outputs.append(y_feat)
	
	model = Model(inputs=[src_in,pose_in,mask_in,trans_in],outputs=outputs)

	return model

def rnn_helper(param,feat_net=None):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	seq_len = param['seq_len']
	batch_size = param['batch_size']

	skip1 = Input(batch_shape=(batch_size,1,128,128,33+28))
	skip2 = Input(batch_shape=(batch_size,1,64,64,128))
	skip3 = Input(batch_shape=(batch_size,1,32,32,256))
	skip4 = Input(batch_shape=(batch_size,1,16,16,256))
	embedding = Input(batch_shape=(batch_size,1,8,8,256))	

	x = myConv(embedding,256,td=True,strides=2,ks=3)		
	x = TimeDistributed(Flatten())(x)
	x = TimeDistributed(Dense(512))(x)	
	x = SimpleRNN(512,activation='relu',kernel_initializer='he_normal',
				  return_sequences=True,stateful=True)(x)

	x = TimeDistributed(Dense(4*4*256))(x)
	x = TimeDistributed(Reshape((4,4,256)))(x)

	x = TimeDistributed(UpSampling2D())(x)
	x = myConv(x,256,ks=3,td=True)	
	x = TimeDistributed(UpSampling2D())(x)
	x = concatenate([x,skip4])
	x = myConv(x,256,ks=3,td=True)
	x = TimeDistributed(UpSampling2D())(x)
	x = concatenate([x,skip3])
	x = myConv(x,256,td=True)
	x = TimeDistributed(UpSampling2D())(x)
	x = concatenate([x,skip2])
	x = myConv(x,128,td=True)
	x = TimeDistributed(UpSampling2D())(x)
	x = concatenate([x,skip1])
	x = myConv(x,64,td=True)
	y = myConv(x,3,activation='tanh',td=True)

	outputs = [y]

	if(feat_net is not None):		
		y = TimeDistributed(Lambda(vgg_preprocess))(y)
		y_feat = TimeDistributed(feat_net)(y)
		outputs.append(y_feat)

	model = Model(inputs=[skip1,skip2,skip3,skip4,embedding],outputs=outputs)

	return model

def make_rnn_from_single(params,single_weights_filename,rnn_weights_filename=None):

	vgg_model = VGG19(weights='imagenet',input_shape=(128,128,3),include_top=False)
	make_trainable(vgg_model,False)

	single_net = network_warp(params,vgg_model)
	single_net.load_weights(single_weights_filename)

	output_names = ['concatenate_2','conv2d_8','conv2d_9','conv2d_10','conv2d_12']
	outputs = []
	for j in output_names:
		outputs.append(single_net.get_layer(j).output)

	single_net = Model(single_net.inputs,outputs)	
	make_trainable(single_net,False)

	rnn_net = rnn_helper(params,vgg_model)			
	if(rnn_weights_filename):
		rnn_net.load_weights(rnn_weights_filename)

	return single_net,rnn_net,vgg_model

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


'''
def switch2(arg):
	a = tf.multiply(arg[0][:,0:512],tf.tile(tf.expand_dims(arg[1][:,0],1),[1,512]))
	b = tf.multiply(arg[0][:,512:],tf.tile(tf.expand_dims(arg[1][:,1],1), [1,512]))	
	return tf.concat([a,b],axis=1)

	x = Lambda(switch2)([x,class_tgt])
'''

'''

def makeWarpedFeatureStack(args):	
	mask = args[0]
	src_in = args[1]
	trans_in = args[2]

	for i in xrange(11):
		mask_i = K.repeat_elements(tf.expand_dims(mask[:,:,:,i],3),10,3)	
		src_masked =  tf.multiply(mask_i,src_in[:,:,:,10*i:10*(i+1)])
		warp_i = affineWarp(src_masked,trans_in[:,:,:,i])
		if(i == 0):
			warps = warp_i
		else:
			warps = tf.concat([warps,warp_i],3)

	return warps
'''
