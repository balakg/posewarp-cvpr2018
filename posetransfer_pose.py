import tensorflow as tf
import os
import numpy as np
import sys
from keras import backend as K
import cv2
import datareader
import preprocess
import networks
import scipy.io as sio
from keras.callbacks import History
from keras.models import load_model,Model

n_joints = 14
batch_size = 8
IMG_HEIGHT = 256
IMG_WIDTH = 256
training_iters = 500000
gpu = '/gpu:3'
test_interval = 100
save_interval = 5000

#Dataset specific vars
n_test_vids = 13
vid_pth = '../../datasets/golfswinghd/videos/'
info_pth = '../../datasets/golfswinghd/videoinfo/'
n_end_remove = 5
img_sfx = '.jpg'
n_train_examples = 200000
n_test_examples = batch_size*2

'''
n_test_vids = 7
vid_pth = '../datasets/weightlifting/videos/Men'
info_pth = '../datasets/weightlifting/videoinfo/'
t_win = 50
n_end_remove = 2
img_sfx = '.png'
'''

def train():	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True


	ex_train,ex_test = datareader.readExamples_transfer(
		vid_pth,info_pth,n_test_vids,n_end_remove,img_sfx,n_train_examples,n_test_examples)

	queue_train = tf.train.slice_input_producer(ex_train,shuffle=True)
	queue_test = tf.train.slice_input_producer(ex_test,shuffle=True)

	param = {}
	stride = 4
	param['crop_size_x'] = 256
	param['crop_size_y'] = 256
	param['target_dist'] = 1.171
	param['scale_max'] = 1.05
	param['scale_min'] = 0.95
	param['stride'] = stride
	param['sigma'] = 7


	hist = History()
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())	
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		
		with tf.device(gpu):
			model_gen = networks.network1(n_joints,IMG_HEIGHT,IMG_WIDTH,3,stride)
			model_pose = load_model('../poseresults/networks/network1/35000.h5')
			model = networks.poseDiscriminatorNet(model_gen,model_pose,IMG_HEIGHT,IMG_WIDTH,3,stride,n_joints)
			#model.summary()



		X_img = np.zeros((batch_size, IMG_HEIGHT,IMG_WIDTH,3))
		X_pose = np.zeros((batch_size, IMG_HEIGHT/stride,IMG_WIDTH/stride,n_joints*2))
		X_ctr = np.zeros((batch_size, IMG_HEIGHT/stride,IMG_WIDTH/stride,1))
		Y = np.zeros((batch_size, IMG_HEIGHT,IMG_WIDTH,3))

		step = 0
    
		while step  < training_iters:
			for i in xrange(batch_size):
				ex = sess.run(queue_train)		
				I0,I1,H0,H1,M = preprocess.makeInputOutputPair(ex,param)
				X_img[i,:,:,:] = I0
				X_pose[i,:,:,:] = np.concatenate((H0,H1),axis=2)
				X_ctr[i,:,:,:] = M
				Y[i,:,:,:] = I1
		
			with tf.device(gpu):
				model.fit([X_img,X_pose,X_ctr],[Y,X_pose[:,:,:,n_joints:]],batch_size=batch_size,
				shuffle=False,epochs=1,verbose=0,callbacks=[hist])
				train_loss = hist.history['loss'][0]

			print "0," + str(train_loss)
			sys.stdout.flush()
	
			if(step % test_interval == 0):

				n_batches = 2
				test_loss = 0
	
				for j in xrange(n_batches):
					for i in xrange(batch_size):
						ex = sess.run(queue_test)
				
						I0,I1,H0,H1,M = preprocess.makeInputOutputPair(ex,param)
						X_img[i,:,:,:] = I0
						X_pose[i,:,:,:] = np.concatenate((H0,H1),axis=2)
						X_ctr[i,:,:,:] = M
						Y[i,:,:,:] = I1
	
					predy,pred_pose = model.predict([X_img,X_pose,X_ctr],batch_size=batch_size,verbose=0)
					d = predy - Y
					test_loss += np.sum(d**2)/(batch_size)
	
				test_loss /= n_batches
				print "1," + str(test_loss/(IMG_HEIGHT*IMG_WIDTH*3))
				sys.stdout.flush()

				sio.savemat('../results/outputs/network_pose2/' + str(step) + '.mat',
                {'X_img': X_img,'Y': Y, 'Ypred': predy})	
	
			if(step % save_interval==0 and step > 0):
				model.save('../results/networks/network_pose2/' + str(step) + '.h5')			

			step += 1	
		
		coord.request_stop()
		coord.join(threads)
		sess.close()

if __name__ == "__main__":
	train()

