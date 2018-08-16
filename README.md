# pose warping

Code for our CVPR 2018 paper: "Synthesizing Images of Humans in Unseen Poses"

Link to paper: http://openaccess.thecvf.com/content_cvpr_2018/papers/Balakrishnan_Synthesizing_Images_of_CVPR_2018_paper.pdf

Run posewarp_train.py first to train a model using vgg loss. 
Then, run posewarp_gan_train.py to refine the model weights further.
network.py has our architecture. 

One train and test video is provided in the data folder
as an example of the data format used by our code. More details coming soon!
