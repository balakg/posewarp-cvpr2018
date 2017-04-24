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
test_save_interval = 200
model_save_interval = 5000

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
n_train_examples = 100000
n_test_examples = 1000

def train():	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	ex_train,ex_test = datareader.makeTransferExampleList(
		vid_pth,info_pth,n_test_vids,n_end_remove,img_sfx,n_train_examples,n_test_examples)

	train_feed = preprocess.transferExampleGenerator(ex_train,batch_size,param)
	test_feed = preprocess.transferExampleGenerator(ex_test,batch_size,param)

	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			model_gen = networks.network1(param)
			model_pose = load_model('../poseresults/networks/network2/20000.h5') 
			model_pose_feat = Model(model_pose.input, model_pose.get_layer('dense2').output)
			model = networks.poseDiscriminatorNet(model_gen,model_pose_feat,param,0.005)

		step = 0	
		while(True):
			X_img,X_pose,X_tgt = next(train_feed)			
			X_feat_tgt = model_pose_feat.predict(X_tgt,batch_size=batch_size) 

			with tf.device(gpu):
				train_loss = model.train_on_batch([X_img,X_pose],[X_tgt,X_feat_tgt])

			print "0," + str(train_loss[0]) + "," + str(train_loss[1]) + "," + str(train_loss[2])
			sys.stdout.flush()
	
			if(step % test_interval == 0):
				n_batches = 8
	
				test_loss = np.array([0.0,0.0,0.0])		
				for j in xrange(n_batches):	
					X_img,X_pose,X_tgt = next(test_feed)
					X_feat_tgt = model_pose_feat.predict(X_tgt,batch_size=batch_size)
					test_loss_j = model.test_on_batch([X_img,X_pose], [X_tgt,X_feat_tgt])
					test_loss += np.array(test_loss_j)
	
				test_loss /= (n_batches) #*param['IMG_HEIGHT']*param['IMG_WIDTH']*3)
				print "1," + str(test_loss[0]) + "," + str(test_loss[1]) + "," + str(test_loss[2])
				sys.stdout.flush()


			if(step % test_save_interval==0):
				X_img,X_pose,X_tgt = next(test_feed)
				pred_val = model.predict([X_img,X_pose])
				X_pred = pred_val[0]
				feat = pred_val[1]
		
				sio.savemat('../results/outputs/network_pose2/' + str(step) + '.mat',
         		{'X_img': X_img,'X_tgt': X_tgt, 'pred': X_pred, 'feat': feat})	

	
			if(step % model_save_interval==0): 
				model.save('../results/networks/network_pose2/' + str(step) + '.h5')			

			step += 1	


if __name__ == "__main__":
	train()

