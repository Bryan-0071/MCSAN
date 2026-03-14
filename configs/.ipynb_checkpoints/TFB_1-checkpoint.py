# 采用TFB方法，但是只有一层

class args():

	# training args
	epochs = 20 #"number of training epochs, default is 10"
	batch_size = 4 #"batch size for training, default is 4"
	dataset_ir = './LLVIP/infrared/train'
	dataset_vi = './LLVIP/visible/train'
	feature_size = (128, 128)
	GPU = '2,3'
	HEIGHT = 128
	WIDTH = 128
	nb_filter = [64, 112, 160, 208, 256]
	save_fusion_model = "./models/train/transformer_sal21/"
	save_loss_dir = './models/train/transformer_sal21/'

	image_size = 128 #"size of training images, default is 256 X 256"
	device = [2,3] #"set it to 1 for running on GPU, 0 for CPU"
	seed = 42 #"random seed for training"

	lr = 1e-5 #"learning rate, default is 0.001"
	log_interval = 10 #"number of images after which the training loss is logged, default is 500"
	resume_fusion_model = None
	# nest net model
	resume_nestfuse = './models/nestfuse/nestfuse_gray_1e2.model'
	fusion_model = './models/transformer/'