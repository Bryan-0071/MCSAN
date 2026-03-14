
class args():

	# training args
	epochs = 100 
	batch_size = 64 

	feature_size = (128, 128)
	GPU = '2,3'
	HEIGHT = 128
	WIDTH = 128
	nb_filter = [64, 128, 256, 512, 768]
	mamba_num_layers = [2,2,3,2]

	image_size = 128 
	seed = 42 #"random seed for training"

	lr = 1e-3 #"learning rate, default is 0.001"
	wd = 0.01 #"weight decay, default is 0.01"

	data_dir = '/home/ec2-user/code/'
	train_list = '/home/ec2-user/code/train.txt'
	vaild_list = '/home/ec2-user/code/valid.txt'
	test_list = '/home/ec2-user/code/test.txt'
	num_steps_stop = 5000
	num_workers = 4