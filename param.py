def getParam(dataset):
	param = {}
	dn = 2 #downsample images from 256 to 128 for now
	param['IMG_HEIGHT'] = 256/dn 
	param['IMG_WIDTH'] = 256/dn
	param['obj_scale_factor'] = 1.171*200.0/dn
	param['scale_max'] = 1.05
	param['scale_min'] = 0.95
	param['max_rotate_degree'] = 10
	param['max_sat_factor'] = 0.05
	param['max_px_shift'] = 10/dn
	param['posemap_downsample'] = 4/dn 
	param['sigma_joint'] = 7.0/dn
	param['n_joints'] = 14

	param['batch_size'] = 8
	param['test_interval'] = 100
	param['test_save_interval'] = 500
	param['model_save_interval'] = 5000

	param['project_dir'] = '/afs/csail.mit.edu/u/b/balakg/pose/pose2image'

	if(dataset == 'golfswinghd'):
		param['n_test_vids'] = 13
		param['vid_pth'] = '../../datasets/golfswinghd/videos/'
		param['info_pth'] = '../../datasets/golfswinghd/videoinfo/'
		param['img_sfx'] = '.jpg'
		param['n_train_examples'] = 1000
		param['n_test_examples'] = 1000

	return param
