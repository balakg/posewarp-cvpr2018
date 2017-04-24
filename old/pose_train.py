import tensorflow as tf
import os
import numpy as np
import sys
import cv2
import datareader
import preprocess
import networks
import scipy.io as sio
from keras.models import load_model,Model
from keras.optimizers import Adam

batch_size = 8
gpu = '/gpu:0'
test_interval = 200
save_interval = 5000

param = {}
param['IMG_HEIGHT'] = 256
param['IMG_WIDTH'] = 256
param['target_dist'] = 1.171
param['scale_max'] = 1.05
param['scale_min'] = 0.95
param['posemap_downsample'] = 4
param['sigma'] = 7
param['n_joints'] = 14

n_test_vids = 13
vid_pth = '../../datasets/golfswinghd/videos/'
info_pth = '../../datasets/golfswinghd/videoinfo/'
n_end_remove = 5
img_sfx = '.jpg'

def train():	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	ex_train,ex_test = datareader.makePoseExampleList(
		vid_pth,info_pth,n_test_vids,n_end_remove,img_sfx)

	train_feed = preprocess.poseExampleGenerator(ex_train,batch_size,param)
	test_feed = preprocess.poseExampleGenerator(ex_test,batch_size,param)

	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			#model = networks.posePredictor2(param)
			model = load_model('../poseresults/networks/network2/10000.h5')	

		step = 10001	
		while(True):
			X_img,X_pose = next(train_feed)			
			#p = model.predict([X_img,X_pose,V],batch_size=batch_size)
			#sio.savemat('test.mat', {'X_img': X_img,'X_tgt': X_tgt, 'p': p, 'V': V})
			#return

			with tf.device(gpu):
				train_loss = model.train_on_batch(X_img,X_pose)

			print "0," + str(train_loss)
			sys.stdout.flush()
	
			if(step % test_interval == 0):
				n_batches = 8
	
				test_loss = 0		
				for j in xrange(n_batches):	
					X_img,X_pose = next(test_feed)
					pred_val = model.predict(X_img,batch_size=batch_size)
					test_loss += np.sum((pred_val-X_pose)**2)/(batch_size)
	
				test_loss /= (n_batches*28.0) #(n_batches*param['IMG_HEIGHT']*param['IMG_WIDTH']*3)
				print "1," + str(test_loss)
				sys.stdout.flush()

				if(step % 600 == 0):
					sio.savemat('../poseresults/outputs/network2/' + str(step) + '.mat',
         			{'X_img': X_img,'X_pose': X_pose, 'pred': pred_val})	
	
			if(step % save_interval==0): # and step > 0):
				model.save('../poseresults/networks/network2/' + str(step) + '.h5')			

			step += 1	


if __name__ == "__main__":
	train()

