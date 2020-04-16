import numpy as np
import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description = 'Signature Matrix Generator')
parser.add_argument('--step_max', type = int, default = 5,
				   help = 'maximum step in ConvLSTM')
parser.add_argument('--gap_time', type = int, default = 10, # tride width...
				   help = 'gap time between each segment')
parser.add_argument('--win_size', type = int, default = [10, 30, 60],
				   help = 'window size of each segment')
parser.add_argument('--min_time', type = int, default = 0,
				   help = 'minimum time point')
parser.add_argument('--max_time', type = int, default = 20000,
				   help = 'maximum time point')
parser.add_argument('--train_start_point',  type = int, default = 0,
						help = 'train start point')
parser.add_argument('--train_end_point',  type = int, default = 8000,
						help = 'train end point')
parser.add_argument('--test_start_point',  type = int, default = 8000,
						help = 'test start point')
parser.add_argument('--test_end_point',  type = int, default = 20000,
						help = 'test end point')
parser.add_argument('--raw_data_path', type = str, default = '../data/synthetic_data_with_anomaly-s-1.csv',
				   help='path to load raw data')
parser.add_argument('--save_data_path', type = str, default = '../data/',
				   help='path to save data')

args = parser.parse_args()
print(args)

step_max = args.step_max
min_time = args.min_time
max_time = args.max_time
gap_time = args.gap_time
win_size = args.win_size

train_start = args.train_start_point
train_end = args.train_end_point
test_start = args.test_start_point
test_end = args.test_end_point

raw_data_path = args.raw_data_path
save_data_path = args.save_data_path

matrix_data_path = save_data_path + "matrix_data/"
if not os.path.exists(matrix_data_path):
	os.makedirs(matrix_data_path)


def generate_signature_matrix_node():
	data = np.array(pd.read_csv(raw_data_path, header = None), dtype=np.float64)
	sensor_n = data.shape[0]

	# min-max normalization
	max_value = np.max(data, axis=1)
	min_value = np.min(data, axis=1)
	data = (np.transpose(data) - min_value)/(max_value - min_value + 1e-6)

	# std normalization
	# data = np.nan_to_num(data)
	# data_mean = np.mean(data, axis = 0)
	# data_std = np.std(data, axis = 0)
	# data = np.transpose(data) - data_mean
	# data = data / (data_std + 1e-5)

	data = np.transpose(data)

	#multi-scale signature matix generation
	for w in range(len(win_size)):
		matrix_all = []
		win = win_size[w]
		print ("generating signature with window " + str(win) + "...")
		for t in range(min_time, max_time, gap_time):
			matrix_t = np.zeros((sensor_n, sensor_n))
			if t >= 60:
				for i in range(sensor_n):
					for j in range(i, sensor_n):
						matrix_t[i][j] = np.inner(data[i, t - win:t], data[j, t - win:t])/(win) # rescale by win
						matrix_t[j][i] = matrix_t[i][j]
			matrix_all.append(matrix_t)

		path_temp = matrix_data_path + "matrix_win_" + str(win)

		np.save(path_temp, matrix_all)
		del matrix_all[:]

	print ("matrix generation finish!")


def generate_train_test_data():
	#data sample generation
	print ("generating train/test data samples...")

	train_data_path = matrix_data_path + "train_data/"
	if not os.path.exists(train_data_path):
		os.makedirs(train_data_path)
	test_data_path = matrix_data_path + "test_data/"
	if not os.path.exists(test_data_path):
		os.makedirs(test_data_path)

	data_all = []
	for w in range(len(win_size)):
		path_temp = matrix_data_path + "matrix_win_" + str(win_size[w]) + ".npy"
		data_all.append(np.load(path_temp))

	train_test_time = [[train_start, train_end], [test_start, test_end]]
	for i in range(len(train_test_time)):
		for data_id in range(int(train_test_time[i][0]/gap_time), int(train_test_time[i][1]/gap_time)):
			step_multi_matrix = []
			for step_id in range(step_max, 0, -1):
				multi_matrix = []
				for i in range(len(win_size)):
					multi_matrix.append(data_all[i][data_id - step_id])
				step_multi_matrix.append(multi_matrix)

			if data_id >= (train_start/gap_time + win_size[-1]/gap_time + step_max) and data_id < (train_end/gap_time): # remove start points with invalid value
				path_temp = os.path.join(train_data_path, 'train_data_' + str(data_id))
				np.save(path_temp, step_multi_matrix)
			elif data_id >= (test_start/gap_time) and data_id < (test_end/gap_time):
				path_temp = os.path.join(test_data_path, 'test_data_' + str(data_id))
				np.save(path_temp, step_multi_matrix)

			del step_multi_matrix[:]

	print ("train/test data generation finish!")


if __name__ == '__main__':
	generate_signature_matrix_node()

	generate_train_test_data()
