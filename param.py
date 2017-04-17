def getParam():
	param = {}
	param['IMG_HEIGHT'] = 256 
	param['IMG_WIDTH'] = 256 
	param['target_dist'] = 1.171
	param['scale_max'] = 1.05
	param['scale_min'] = 0.95
	param['max_rotate_degree'] = 10
	param['max_sat_factor'] = 0.05
	param['max_px_shift'] = 10
	param['posemap_downsample'] = 4 
	param['sigma'] = 7 
	param['n_joints'] = 14

	return param
