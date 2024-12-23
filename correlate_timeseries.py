import sys
import os
import pandas as pd
import random
from tslearn.clustering import KShape


input_dir = "Input/CT/"
input_data_dir = input_dir + "Data/"

output_dir = "Output/CT/"
output_data_dir = output_dir + "Data/"


def read_time_series_windows():
	data = []

	for file in os.listdir(input_data_dir):
		data.append(pd.read_csv(input_data_dir + file))
	
	return data

def cluster_ts(time_series_windows, n_ts_clusters):

	clustered_time_series_windows = []
	columns = list(time_series_windows[0].columns)

	cluster_columns = []
	for column in columns:
		if column.find("J") == -1:
			cluster_columns.append(column)

	
	time_series_windows_no_cluster = []
	for time_series_window in time_series_windows:
		time_series_windows_no_cluster.append(time_series_window.drop(labels=cluster_columns, axis=1))

	ks = KShape(n_clusters=n_ts_clusters, n_init=1).fit(time_series_windows_no_cluster)
	ks = list(ks.predict(time_series_windows_no_cluster))
	
	
	'''
	ks = KShape(n_clusters=n_ts_clusters, n_init=1, random_state=0).fit(time_series_windows)
	ks = list(ks.predict(time_series_windows))
	'''

	least_frequent_cluster = min(ks,key=ks.count)

	for idx, ts_window_cluster in enumerate(ks):
		if ks[idx] == least_frequent_cluster:
			clustered_time_series_windows.append(time_series_windows[idx])

	return clustered_time_series_windows

def write_time_series_windows(clustered_time_series_windows):

	for idx,ts_window in enumerate(clustered_time_series_windows):
		ts_window.to_csv(output_data_dir + "WNDW_" + str(idx) + ".csv", index = False)

	return None
	

try:
	n_ts_clusters = int(sys.argv[1])
except:
	print("Enter the right number of input arguments.")
	sys.exit()

time_series_windows = read_time_series_windows()
clustered_time_series_windows = cluster_ts(time_series_windows, n_ts_clusters)
write_time_series_windows(clustered_time_series_windows)

'''

distance, paths = dtw.warping_paths(time_series_a, time_series_b, use_c=False)
best_path = dtw.best_path(paths)

data = read_data()

new_data = []

nr_a_windows_to_retain = int(len(data["A"])*tp_percentage)
nr_n_windows_to_retain = int(len(data["N"])*fp_percentage)

for i in range(0,nr_a_windows_to_retain):
	idx = random.randint(0, len(data["A"])-1)
	sampled_window = data["A"].pop(idx)
	new_data.append(sampled_window)

for i in range(0,nr_n_windows_to_retain):
	idx = random.randint(0, len(data["N"])-1)
	sampled_window = data["N"].pop(idx)
	new_data.append(sampled_window)

write_data(new_data)

'''






