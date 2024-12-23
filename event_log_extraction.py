import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
import pandas as pd
import os

from random import seed
from random import random
from random import randint

import math

from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

input_dir = "Input/ELE/"
input_data_dir = input_dir + "Data/"

output_dir = "Output/ELE/"
output_eventlog_dir = output_dir + "EventLog/"

def read_data():
	windows = []

	for window_file in os.listdir(input_data_dir):
		windows.append(pd.read_csv(input_data_dir + window_file))
		
	return windows
	
def extract_state_transitions(data, sensor_types, synthetic_data):
	state_transitions = []
	cluster_columns = []
	if synthetic_data == 0:
		if "S" in sensor_types:
			cluster_columns.append("CS")
		for i in range(1,7+1):
			if "J"+str(i) in sensor_types:
				cluster_columns.append("C"+str(i))

		current_state = {}
		for idx,sample in data.iterrows():
			if len(current_state) == 0:
				for cluster in cluster_columns:
					current_state[cluster] = sample[cluster]
			else:
				next_state = {}
				for cluster in cluster_columns:
					next_state[cluster] = sample[cluster]
				for cluster in next_state:
					if next_state[cluster] != current_state[cluster]:
						if cluster == "CS":
							new_state_transition = "S-" + str(int(current_state[cluster])) + "_" + str(int(next_state[cluster]))
						else:
							new_state_transition = "J" + str(cluster[1]) + "-" + str(int(current_state[cluster])) + "_" + str(int(next_state[cluster]))
						state_transitions.append([new_state_transition,idx])
						current_state[cluster] = next_state[cluster]
	elif synthetic_data == 1:
		cluster_columns = "C"
		current_state = {}
		for idx,sample in data.iterrows():
			if len(current_state) == 0:
				for cluster in cluster_columns:
					current_state[cluster] = sample[cluster]
			else:
				next_state = {}
				for cluster in cluster_columns:
					next_state[cluster] = sample[cluster]
				for cluster in next_state:
					if next_state[cluster] != current_state[cluster]:
						new_state_transition = "C" + "-" + str(int(current_state[cluster])) + "_" + str(int(next_state[cluster]))
						state_transitions.append([new_state_transition,idx])
						current_state[cluster] = next_state[cluster]

	return state_transitions	

def build_event_log(windows_state_transitions):
	
	event_log = []
	for idx,window_state_transitions in enumerate(windows_state_transitions):
		caseid = idx
		for state_transitions in window_state_transitions:
			event_timestamp = timestamp_builder(state_transitions[1])
			state_transition = state_transitions[0]
			event = [caseid, state_transition, event_timestamp]
			event_log.append(event)
	'''
	for state_transition in windows_state_transitions:
		caseid = 0
		event_timestamp = timestamp_builder(state_transition[1])
		state_transition = state_transition[0]
		event = [caseid, state_transition, event_timestamp]
		event_log.append(event)
	'''
	event_log = pd.DataFrame(event_log, columns=['CaseID', 'Event', 'Timestamp'])
	event_log.rename(columns={'Event': 'concept:name'}, inplace=True)
	event_log.rename(columns={'Timestamp': 'time:timestamp'}, inplace=True)
	event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
	parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'CaseID'}
	event_log = log_converter.apply(event_log, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)
		
	return event_log

def timestamp_builder(number):
	
	ss = number
	mm, ss = divmod(ss, 60)
	hh, mm = divmod(mm, 60)
	ignore, hh = divmod(hh, 24)
	
	ss = ss%60
	mm = mm%60
	hh = hh%24
	
	return "1900-01-01T"+str(hh)+":"+str(mm)+":"+str(ss)	
	
def save_log(event_log):

	xes_exporter.apply(event_log, output_eventlog_dir + "EL.xes")

	return None		
	
try:
	synthetic_data = int(sys.argv[1])
	sensor_types = []
	if synthetic_data == 0:
		n_sensor_types = int(sys.argv[2])
		for i in range(1, n_sensor_types+1):
			sensor_types.append(sys.argv[2+i])
except:
	print("Enter the right number of input arguments.")
	sys.exit()	
	
windows = read_data()

windows_state_transitions = []
for window in windows:
	windows_state_transitions.append(extract_state_transitions(window, sensor_types, synthetic_data))
event_log = build_event_log(windows_state_transitions)

save_log(event_log)