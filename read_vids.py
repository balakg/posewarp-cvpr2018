import os
import sys
import glob

base_data_dir = '/afs/csail.mit.edu/u/b/balakg/pose/datasets/golfswinghd/frames/*' 
vid_names = glob.glob(base_data_dir)

for i in vid_names:
	print i
