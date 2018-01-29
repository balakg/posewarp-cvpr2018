import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D,Dense,Activation,Input,UpSampling2D,Dropout
from keras.layers import concatenate,Flatten,Reshape,Lambda
from keras.layers import LeakyReLU
from keras.applications.vgg19 import VGG19
from keras.initializers import RandomNormal
from keras.optimizers import Adam,RMSprop
import keras

def myConv(x_in,nf,ks=3,strides=1,activation='lrelu',ki='he_normal',name=None,dropout=False):

	x_out = Conv2D(nf,kernel_size=ks, padding='same',kernel_initializer=ki,strides=strides)(x_in)

	if(dropout):
		x_out = Dropout(0.2)(x_out)

	if(name):
		if(activation == 'lrelu'):
			x_out = LeakyReLU(0.2,name=name)(x_out)
		elif(activation != 'none'):
			x_out = Activation(activation,name=name)(x_out)
	else:
		if(activation == 'lrelu'):
			x_out = LeakyReLU(0.2)(x_out)
		elif(activation != 'none'):
			x_out = Activation(activation)(x_out)

	return x_out


def myDense(x_in,nf,activation='relu',ki='he_normal',dropout=False):
	if(dropout):
		x_in = Dropout(0.5)(x_in)

	x_out = Dense(nf,activation=activation,kernel_initializer=ki)(x_in)
	return x_out

def vggLoss(feat_net,feat_weights):
	def loss_fcn(y_true,y_pred):
		y_true_feat = feat_net(Lambda(vgg_preprocess)(y_true))
		y_pred_feat = feat_net(Lambda(vgg_preprocess)(y_pred))

		loss = []
		for j in xrange(12):
			std = feat_weights[str(j)][1]+0.1
			std = tf.expand_dims(tf.expand_dims(tf.expand_dims(std,0),0),0)
			d = tf.subtract(y_true_feat[j],y_pred_feat[j])
			loss_j = tf.reduce_mean(tf.abs(tf.divide(d,std)))
		
			if(j == 0):
				loss = loss_j
			else:
				loss = tf.add(loss,loss_j)
		return loss/12.0

	return loss_fcn


def make_trainable(net,val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

def discriminator(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	x_tgt = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	x_src_pose = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))
	x_tgt_pose = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))

	x = myConv(x_tgt,64,ks=5,strides=2) #128
	x = concatenate([x,x_src_pose,x_tgt_pose])
	x = myConv(x,128,ks=5,strides=2) #64
	x = myConv(x,256,strides=2) #32
	x = myConv(x,256,strides=2) #16
	x = myConv(x,256,strides=2) #8
	x = myConv(x,256) #8

	x = Flatten()(x)

	x = myDense(x,256)
	x = myDense(x,256)
	y = myDense(x,1,activation='sigmoid')
	#y = myDense(x,1,activation='linear') #for wgan

	model = Model(inputs=[x_tgt,x_src_pose,x_tgt_pose],outputs=y, name='discriminator')
	return model

def wass(y_true,y_pred):
	return tf.reduce_mean(y_true*y_pred)

def gan(generator,discriminator,param,feat_net,feat_weights,disc_loss,lr):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_src = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))
	pose_tgt = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))
	mask_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))

	make_trainable(discriminator, False)
	#y_gen = generator([src_in,pose_src,pose_tgt])
	y_gen = generator([src_in,pose_src,pose_tgt,mask_in,trans_in])
	y_class = discriminator([y_gen,pose_src,pose_tgt])
	
	#gan = Model(inputs=[src_in,pose_src,pose_tgt],outputs=[y_gen,y_class], name='gan')
	gan = Model(inputs=[src_in,pose_src,pose_tgt,mask_in,trans_in],outputs=[y_gen,y_class], name='gan')

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


def makeWarpedStack(args):	
	mask = args[0]
	src_in = args[1]
	trans_in = args[2]

	ctr = 0
	for i in xrange(11):
		mask_i = K.repeat_elements(tf.expand_dims(mask[:,:,:,i],3),3,3)	
		src_masked = tf.multiply(mask_i,src_in)

		if(i == 0):
			warps = src_masked	
			src_masked = tf.add(src_masked,tf.multiply(1-mask_i,
						 tf.random_uniform(tf.shape(src_in),-0.003, 0.003)))
		else:
			warp_i = affineWarp(src_masked, trans_in[:,:,:,ctr])
			warps = tf.concat([warps,warp_i],3)
		
		ctr += 1

	return warps


def vgg_preprocess(arg):
	z = 255.0 * (arg+1.0)/2.0
	r = z[:,:,:,0] - 103.939
	g = z[:,:,:,1] - 116.779
	b = z[:,:,:,2] - 123.68
	return tf.stack([r,g,b],axis=3)

def unet(x_in,pose_in,nf_enc,nf_dec):

	x0 = myConv(x_in,nf_enc[0],ks=7) #256
	x1 = myConv(x0,nf_enc[1],strides=2)#128
	x2 = concatenate([x1,pose_in])
	x3 = myConv(x2,nf_enc[2])
	x4 = myConv(x3,nf_enc[3],strides=2)#64
	x5 = myConv(x4,nf_enc[4])
	x6 = myConv(x5,nf_enc[5],strides=2)#32
	x7 = myConv(x6,nf_enc[6])
	x8 = myConv(x7,nf_enc[7],strides=2)#16
	x9 = myConv(x8,nf_enc[8])
	x10 = myConv(x9,nf_enc[9],strides=2)#8
	x = myConv(x10,nf_enc[10])

	skips = [x9,x7,x5,x3,x0]
	for i in xrange(5):
		x = UpSampling2D()(x)	
		x = concatenate([x,skips[i]])

		do_dropout = False
		if i < 2:
			do_dropout=True

		x = myConv(x,nf_dec[i],dropout=do_dropout)	
	
	return x


def network_fgbg(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_src = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))
	pose_tgt = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))
	src_mask_prior = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))

	#1. FG/BG separation
	x = unet(src_in,pose_src,[64]*2+[128]*9,[128]*4+[32])
	src_mask_delta = myConv(x,11,activation='linear')
	src_mask = keras.layers.add([src_mask_delta,src_mask_prior])
	src_mask = Activation('softmax',name='mask_src')(src_mask)

	#2. Separate into fg limbs and background
	warped_stack = Lambda(makeWarpedStack)([src_mask,src_in,trans_in])
	fg_stack = Lambda(lambda arg: arg[:,:,:,3:],output_shape=(256,256,30),
				name='fg_stack')(warped_stack)	
	bg_src = Lambda(lambda arg: arg[:,:,:,0:3],output_shape=(256,256,3),
				name='bg_src')(warped_stack)
	bg_src_mask = Lambda(lambda arg: tf.expand_dims(arg[:,:,:,0],3))(src_mask)

	#3. BG/FG synthesis	
	x = unet(concatenate([bg_src,bg_src_mask]),pose_src,[64]*2+[128]*9,[128]*4+[64])
	bg_tgt = myConv(x,3,activation='tanh',name='bg_tgt')

	x = unet(fg_stack,pose_tgt,[64]+[128]*3+[256]*7,[256,256,256,128,64])
	fg_tgt = myConv(x,3,activation='tanh',name='fg_tgt')

	fg_mask = myConv(x,1,activation='sigmoid',name='fg_mask_tgt')
	fg_mask = concatenate([fg_mask,fg_mask,fg_mask])
	bg_mask = Lambda(lambda arg: 1-arg)(fg_mask)	

	#5. Merge bg and fg
	fg_tgt = keras.layers.multiply([fg_tgt,fg_mask],name='fg_tgt_masked')
	bg_tgt = keras.layers.multiply([bg_tgt,bg_mask],name='bg_tgt_masked')
	y = keras.layers.add([fg_tgt,bg_tgt])

	model = Model(inputs=[src_in,pose_src,pose_tgt,src_mask_prior,trans_in], outputs=[y])

	return model


def network_pix2pix(param):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_src = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))
	pose_tgt = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn,14))

	x = unet(src_in,concatenate([pose_src,pose_tgt]),[64]+[128]*3+[256]*7,[256,256,256,128,64])
	y = myConv(x,3,activation='tanh')

	model = Model(inputs=[src_in,pose_src,pose_tgt], outputs=[y])
	return model

'''
def resBlock(x_in,nf1,nf2,strides,upsample=False):
	x = Conv2D(nf1,kernel_size=3, padding='same',kernel_initializer='he_normal',activation='relu')(x_in)
	x = Conv2D(nf1,kernel_size=3, padding='same',kernel_initializer='he_normal',activation='relu')(x)
	x = keras.layers.add([x,x_in])
	x = Conv2D(nf2,kernel_size=3, padding='same',kernel_initializer='he_normal',strides=strides)(x)

	if(upsample):
		x = UpSampling2D()(x)

	return x

def g1():

	IA = Input(shape=(256,256,3))
	pA = Input(shape=(256,256,14))
	pB = Input(shape=(256,256,14))

	x_in = concatenate([IA,pA,pB])
	x1 = resBlock(x_in,31,64,strides=1)
	x2 = resBlock(x1,64,128,strides=2) #128
	x3 = resBlock(x2,128,256,strides=2) #64
	x4 = resBlock(x3,256,256,strides=2) #32
	x5 = resBlock(x4,256,256,strides=2) #16
	x6 = resBlock(x5,256,256,strides=2) #8

	#x = Flatten()(x6)
	#x = Dense(64)(x)
	#x = Dense(8*8*256)(x)
	#x = Reshape((8,8,256))(x)

	#x = concatenate([x,x6]) #8
	x = resBlock(x6,256,256,strides=1,upsample=True) #16

	x = concatenate([x,x5]) 
	x = resBlock(x,512,256,strides=1,upsample=True) #32

	x = concatenate([x,x4])
	x = resBlock(x,512,256,strides=1,upsample=True) #64

	x = concatenate([x,x3])
	x = resBlock(x,512,128,strides=1,upsample=True) #128
	
	x = concatenate([x,x2])
	x = resBlock(x,256,64,strides=1,upsample=True) #256

	y = Conv2D(3,kernel_size=3, padding='same',kernel_initializer='he_normal',activation='tanh')(x)	

	model = Model(inputs=[IA,pA,pB],outputs=y)
	return model
'''
