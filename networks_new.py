import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D,Dense,Activation,Input,UpSampling2D
from keras.layers import concatenate,Flatten,Reshape,Lambda
from keras.layers import LeakyReLU,GlobalAveragePooling2D
from keras.applications.vgg19 import VGG19
from keras.optimizers import Adam
import keras

def conv_block(x_in,nf,ks=5,activation='relu',strides=1,ki='he_normal',name=None,dropout=False):

	layer = Conv2D(nf,kernel_size=ks, padding='same',kernel_initializer=ki,strides=strides,activation=activation)

	x_out = layer(x_in)
	x_out = BatchNormalization()(x_out)

	if(dropout):
		X = Dropout(0.5)(x)

	if(activation == 'lrelu'):
		x_out = LeakyReLU(0.2,name=name)(x_out)
	else:
		x_out = Activation(activation,name=name)(x_out)

	return x_out

def myDense(x_in,nf,activation='relu',ki='he_normal'):
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
	x_tgt_pose = Input(shape=(IMG_HEIGHT/pose_dn,IMG_WIDTH/pose_dn, n_joints))

	x_pose = UpSampling2D()(x_tgt_pose)
	x = concatenate([x_src,x_pose])
	x = myConv(x,256,ks=11)
	x = myConv(x,256,ks=5,strides=2)
	x = myConv(x,256,ks=5)
	x = myConv(x,256,ks=5,strides=2)
	x = myConv(x,256,ks=5)
	x = myConv(x,1,ks=5,activation='sigmoid',name='responses')
	y = GlobalAveragePooling2D()(x) 

	model = Model(inputs=[x_src,x_tgt_pose],outputs=y, name='discriminator')
	return model

def gan(generator,discriminator,param,feat_net,feat_weights):

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
	#generator_mask = Model(generator.input, [generator.output, generator.get_layer('fg_tgt_masked').output])

	y_gen = generator([src_in,pose_src,pose_tgt,mask_in,trans_in])
	y_class = discriminator([y_gen,pose_tgt])

	def vggLoss(y_true,y_pred):
		y_true = Lambda(vgg_preprocess)(y_true)
		y_pred = Lambda(vgg_preprocess)(y_pred)
		y_true_feat = feat_net(y_true)
		y_pred_feat = feat_net(y_pred)

		loss = []
		for j in xrange(12):
			std = feat_weights[str(j)][1]
			std = tf.expand_dims(tf.expand_dims(tf.expand_dims(std,0),0),0)

			batch_size = tf.shape(std)[0]
			height = tf.shape(std)[1]
			width = tf.shape(std)[2]
			weights = tf.tile(std, [batch_size,height,width,1])

			d = tf.subtract(y_true_feat[j],y_pred_feat[j])
			d = tf.divide(d,std)
			d = tf.abs(d)
			loss_j = tf.reduce_mean(d)
		
			if(j == 0):
				loss = loss_j
			else:
				loss = tf.add(loss,loss_j)

		return loss/12.0

	gan = Model(inputs=[src_in,pose_src,pose_tgt,mask_in,trans_in],outputs=[y_gen,y_class], name='gan')
	gan.compile(optimizer=Adam(lr=1e-4),loss=[vggLoss, 'binary_crossentropy'], loss_weights=[1.0,0.1])

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
		if(i == 0):
			warps = src_masked
		else:
			warp_i = affineWarp(src_masked,trans_in[:,:,:,i])
			warps = tf.concat([warps,warp_i],3)

	return warps


def vgg_preprocess(arg):
	z = 255.0 * (arg+1.0)/2.0
	r = z[:,:,:,0] - 103.939
	g = z[:,:,:,1] - 116.779
	b = z[:,:,:,2] - 123.68
	return tf.stack([r,g,b],axis=3)


def network_fgbg(param,feat_net=None, feat_weights=None):

	IMG_HEIGHT = param['IMG_HEIGHT']
	IMG_WIDTH = param['IMG_WIDTH']
	n_joints = param['n_joints']
	pose_dn = param['posemap_downsample']

	src_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,3))
	pose_src = Input(shape=(IMG_HEIGHT,IMG_WIDTH,n_joints))
	pose_tgt = Input(shape=(IMG_HEIGHT,IMG_WIDTH,n_joints))
	mask_in = Input(shape=(IMG_HEIGHT,IMG_WIDTH,11))	
	trans_in = Input(shape=(2,3,11))



def conv_block(x_in,nf,ks=5,activation='relu',strides=1,ki='he_normal',name=None,dropout=False):
	x = concatenate([src_in,pose_src])
	
	list_src_enc = [x]
	ctr = 1
	for i in xrange(5):
		list_src_enc.append(conv_block(list_src_enc[-1],128,3,'lrelu',name='c1_' + str(ctr)))
		list_src_enc.append(conv_block(list_src_enc[-1],128,3,'lrelu',2,name='c1_' + str(len(list_src_enc))))

	list_src_enc.append(conv_block(list_src_enc[-1],128,3,'lrelu',name='c1_'+str(len(list_src_enc))))

	x = list_src_enc[-1]
	for i in xrange(5):
		x = UpSampling2D()(x)
		x = concatenate([x,list_src_enc[8-i*2]])
		x = conv_block(x,128,3,'relu',name='c1_'+str(len(list_src_enc)+i+1))


	mask_delta = conv_block(x,11,ks=3,activation='linear',ki='zeros',name=')
	mask = keras.layers.add([mask_delta,mask_in])
	mask = Activation('softmax',name='mask_src')(mask)

	warped_stack = Lambda(makeWarpedStack)([mask,src_in,trans_in])
	fg_stack = Lambda(lambda arg: arg[:,:,:,3:],output_shape=(256,256,30),name = 'fg_stack')(warped_stack)	
	bg_src = Lambda(lambda arg: arg[:,:,:,0:3],output_shape=(256,256,3),name = 'bg_src')(warped_stack)
	bg_src_mask = Lambda(lambda arg: tf.expand_dims(arg[:,:,:,0],3))(mask)
	bg_src = concatenate([bg_src,bg_src_mask])

	x0 = myConv(bg_src,32,ks=11) #256
	x1 = myConv(x0,128,ks=3,strides=2)#128
	x1 = concatenate([x1,pose_src])
	x2 = myConv(x1,128,ks=3)
	x3 = myConv(x2,128,ks=3,strides=2)#64
	x3 = myConv(x3,128,ks=3)
	x4 = myConv(x3,256,ks=3,strides=2)#32
	x4 = myConv(x4,256,ks=3)
	x5 = myConv(x4,256,ks=3,strides=2)#16
	x5 = myConv(x5,256,ks=3)
	x6 = myConv(x5,256,ks=3,strides=2)#8
	x6 = myConv(x6,256,ks=3)
	x7 = myConv(x6,256,ks=3)#8

	x = UpSampling2D()(x7) #16x16x256
	x = concatenate([x,x5])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #32x32x256
	x = concatenate([x,x4])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #64x64x256
	x = concatenate([x,x3])
	x = myConv(x,128,ks=3)
	x = UpSampling2D()(x) #128
	x = concatenate([x,x2])
	x = myConv(x,128,ks=3)
	x = UpSampling2D()(x) #256
	x = concatenate([x,x0])
	x = myConv(x,32,ks=3)
	x = myConv(x,32,ks=11)
	bg_tgt = myConv(x,3,ks=11,activation='tanh',name='bg_tgt')


	x0 = myConv(fg_stack,128,ks=11) #256
	x1 = myConv(x0,128,ks=3,strides=2)#128
	x1 = concatenate([x1,pose_tgt])
	x2 = myConv(x1,128,ks=3)
	x3 = myConv(x2,128,ks=3,strides=2)#64
	x3 = myConv(x3,128,ks=3)
	x4 = myConv(x3,256,ks=3,strides=2)#32
	x4 = myConv(x4,256,ks=3)
	x5 = myConv(x4,256,ks=3,strides=2)#16
	x5 = myConv(x5,256,ks=3)
	x6 = myConv(x5,256,ks=3,strides=2)#8
	x6 = myConv(x6,256,ks=3)
	x7 = myConv(x6,256,ks=3)#8

	x = UpSampling2D()(x7) #16x16x256
	x = concatenate([x,x5])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #32x32x256
	x = concatenate([x,x4])
	x = myConv(x,256,ks=3)
	x = UpSampling2D()(x) #64x64x256
	x = concatenate([x,x3])
	x = myConv(x,128,ks=3)
	x = UpSampling2D()(x) #128x128x128
	x = concatenate([x,x2])
	x = myConv(x,128,ks=3)
	x = UpSampling2D()(x) #256x256
	x = concatenate([x,x0])
	x = myConv(x,32,ks=3)
	x = myConv(x,32,ks=11)

	fg_tgt = myConv(x,3,ks=11,activation='tanh',name='fg_tgt')#256x256x3
	fg_mask = myConv(x,1,ks=11,activation='sigmoid',name='fg_mask_tgt')

	fg_mask = concatenate([fg_mask,fg_mask,fg_mask])
	bg_mask = Lambda(lambda arg: 1-arg)(fg_mask)	

	fg_tgt = keras.layers.multiply([fg_tgt,fg_mask],name='fg_tgt_masked')
	bg_tgt = keras.layers.multiply([bg_tgt,bg_mask])
	y = keras.layers.add([fg_tgt,bg_tgt])

	def vggLoss(y_true,y_pred):
		y_true = Lambda(vgg_preprocess)(y_true)
		y_pred = Lambda(vgg_preprocess)(y_pred)
		y_true_feat = feat_net(y_true)
		y_pred_feat = feat_net(y_pred)

		loss = []
		for j in xrange(12):
			std = feat_weights[str(j)][1]
			std = tf.expand_dims(tf.expand_dims(tf.expand_dims(std,0),0),0)

			batch_size = tf.shape(std)[0]
			height = tf.shape(std)[1]
			width = tf.shape(std)[2]
			weights = tf.tile(std, [batch_size,height,width,1])

			d = tf.subtract(y_true_feat[j],y_pred_feat[j])
			d = tf.divide(d,std)
			#d = tf.multiply(d,d)
			d = tf.abs(d)
			loss_j = tf.reduce_mean(d)
		
			if(j == 0):
				loss = loss_j
			else:
				loss = tf.add(loss,loss_j)

		return loss/12.0


	model = Model(inputs=[src_in,pose_src,pose_tgt,mask_in,trans_in],outputs=[y])
	model.compile(optimizer=Adam(lr=1e-4),loss=[vggLoss])
	
	return model
