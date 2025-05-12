import os

class Parameters():
	def __init__(self):
		self.n_processors = 8
		# Path
		self.data_dir = os.path.join(os.getcwd(), 'KITTI')
		self.image_dir = os.path.join(self.data_dir, 'images')
		self.pose_dir = os.path.join(self.data_dir, 'pose_GT')

		self.train_video = ['00', '01', '02', '05', '08', '09']
		self.valid_video = ['04', '06', '07', '10']
		self.partition = None  # partition videos in 'train_video' to train / valid dataset  #0.8


		# Data Preprocessing
		self.resize_mode = 'rescale'  # choice: 'crop' 'rescale' None
		self.img_w = 608   # original size is about 1226
		self.img_h = 184   # original size is about 370
		self.img_means =  (0.19007764876619865, 0.15170388157131237, 0.10659445665650864)
		self.img_stds =  (0.2610784009469139, 0.25729316928935814, 0.25163823815039915)
		# self.img_means =  (-0.1492917483178474, -0.13016938249129348, -0.13576983730061404)
		# self.img_stds =  (0.31573439925650076, 0.3193686646592632, 0.32300527971493376)
		self.minus_point_5 = True

		self.seq_len = (5, 7)
		#self.seq_len = (2, 7)
		self.sample_times = 3

		# Data info path
		self.train_data_info_path = os.path.join(os.getcwd(), 'datainfo',
												 'train_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(
													 ''.join(self.train_video), ''.join(self.valid_video),
													 self.partition, self.seq_len[0], self.seq_len[1],
													 self.sample_times))

		self.valid_data_info_path = os.path.join(os.getcwd(), 'datainfo',
												 'valid_df_t{}_v{}_p{}_seq{}x{}_sample{}.pickle'.format(
													 ''.join(self.train_video), ''.join(self.valid_video),
													 self.partition, self.seq_len[0], self.seq_len[1],
													 self.sample_times))

		# Model
		self.rnn_hidden_size = 1000
		self.conv_dropout = (0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.5)
		self.rnn_dropout_out = 0.5
		self.rnn_dropout_between = 0   # 0: no dropout
		self.clip = None
		self.batch_norm = True
		# Training
		self.epochs = 250
		self.batch_size = 16 #8
		self.pin_mem = False
		self.optim = {'opt': 'Adagrad', 'lr': 0.0005}
					# Choice:
					# {'opt': 'Adagrad', 'lr': 0.001}
					# {'opt': 'Adam'}
					# {'opt': 'Cosine', 'T': 100 , 'lr': 0.001}

		# Pretrain, Resume training
		self.pretrained_flownet = None
								# Choice:
								# None
								# './pretrained/flownets_bn_EPE2.459.pth.tar'
								# './pretrained/flownets_EPE1.951.pth.tar'
		self.resume = True  # resume training
		self.resume_t_or_v = '.train'

		#self.load_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)
		#self.load_optimizer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer{}'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]), self.resume_t_or_v)

		self.load_model_path = 'models/t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.model.train'
		self.load_optimizer_path = 'models/t000102050809_v04060710_im184x608_s5x7_b8_rnn1000_optAdagrad_lr0.0005.optimizer.train'

		self.record_path = 'records/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.txt'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		self.save_model_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.model'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))
		self.save_optimzer_path = 'models/t{}_v{}_im{}x{}_s{}x{}_b{}_rnn{}_{}.optimizer'.format(''.join(self.train_video), ''.join(self.valid_video), self.img_h, self.img_w, self.seq_len[0], self.seq_len[1], self.batch_size, self.rnn_hidden_size, '_'.join([k+str(v) for k, v in self.optim.items()]))


		if not os.path.isdir(os.path.dirname(self.record_path)):
			os.makedirs(os.path.dirname(self.record_path))
		if not os.path.isdir(os.path.dirname(self.save_model_path)):
			os.makedirs(os.path.dirname(self.save_model_path))
		if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
			os.makedirs(os.path.dirname(self.save_optimzer_path))
		if not os.path.isdir(os.path.dirname(self.train_data_info_path)):
			os.makedirs(os.path.dirname(self.train_data_info_path))

	def print_params(self):
		"""Print all parameters to check if initialized correctly."""
		for key, value in self.__dict__.items():
			print(f"{key}: {value}")

par = Parameters()



# Instantiate and check values
if __name__ == "__main__":
    par = Parameters()
    par.print_params()