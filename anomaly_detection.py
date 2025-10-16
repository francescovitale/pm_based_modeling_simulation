import sys
import os
import pandas as pd
import random


input_dir = "Input/AD/"
input_data_dir = input_dir + "Data/"

output_dir = "Output/AD/"
output_data_dir = output_dir + "Data/"


def read_time_series_windows():
	time_series_windows = {}
	
	for window_type in os.listdir(input_data_dir):
		time_series_windows[window_type] = []
		for window in os.listdir(input_data_dir + window_type):
			time_series_windows[window_type].append(pd.read_csv(input_data_dir + window_type + "/" + window))
	
	return time_series_windows
	
def	build_training_set(time_series_windows, accuracy):
	
	training_set = []
	
	number_of_true_positives = int(len(time_series_windows["A"])*accuracy)
	number_of_false_positives = int(len(time_series_windows["N"])*(1-accuracy))
	
	for i in range(0,number_of_true_positives):
		training_set.append(time_series_windows["A"][i])
	for i in range(0,number_of_false_positives):
		training_set.append(time_series_windows["N"][i])
	
	return training_set

def write_time_series_windows(training_set):

	for idx,ts_window in enumerate(training_set):
		ts_window.to_csv(output_data_dir + "WNDW_" + str(idx) + ".csv", index = False)

	return None
	
try:
	accuracy = float(sys.argv[1])
except:	
	print("Enter the right number of input arguments")
	sys.exit()

time_series_windows = read_time_series_windows()
training_set = build_training_set(time_series_windows, accuracy)
write_time_series_windows(training_set)







