from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

import pm4py
import sys
import os

from itertools import combinations,permutations

import time

input_dir = "Input/PD/"
input_eventlog_dir = input_dir + "EventLog/"

output_dir = "Output/PD/"
output_timing_dir = output_dir + "Timing/"
output_petrinet_dir = output_dir + "PetriNet/"

variant = ""

def read_data():

	centroids = read_centroids()
	event_log = xes_importer.apply(input_eventlog_dir + "EL_TR.xes")

	return event_log, centroids
	
def read_centroids():
	centroids = {}

	file = open(input_dir + "clustering_parameters.txt","r")
	lines = file.readlines()
	for line in lines:
		line = line.replace("\n","")
		line = line.replace(" ","")
		line = line.replace("[","")
		line = line.replace("]","")
		tokens = line.split(":")
		centroid_coordinates = tokens[-1].split(",")
		centroids[tokens[0]] = []
		for centroid_coordinate in centroid_coordinates:
			centroids[tokens[0]].append(float(centroid_coordinate))
	file.close()

	return centroids	
	
def process_discovery(event_log, variant):

	petri_net = {}

	if variant == "im":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log)
	
	elif variant == "imf25":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log, noise_threshold = 0.25)

	elif variant == "imf50":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log, noise_threshold = 0.50)

	elif variant == "imf75":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log, noise_threshold = 0.75)

	elif variant == "imf99":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_inductive(event_log, noise_threshold = 0.99)

	elif variant == "ilp":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=1-0.00)

	elif variant == "ilp25":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=1-0.25)

	elif variant == "ilp50":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=1-0.50)

	elif variant == "ilp75":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=1-0.75)

	elif variant == "ilp99":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=1-0.99)

	elif variant == "alpha":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_alpha(event_log)
		
	elif variant == "hm":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_heuristics(event_log, dependency_threshold=0.75)

	return petri_net

def get_statistics(event_log, centroids, synthetic_data):

	states = list(centroids.keys())
	state_pairs = list(permutations(states, 2))
	average_trace_length = 0
	
	# time of initial state: the time it takes to fire a new state transition starting from the initial state
	times_of_initial_state = {}
	for state in states:
		times_of_initial_state[str(state)] = []

	# time of state: the time it takes to fire a new state transition when in a state reached by a given transition.
	times_of_states = {}
	for pair in state_pairs:
		if synthetic_data == 0:
			times_of_states["J2-" + str(pair[0]) + "_" + str(pair[1])] = []
		else:
			times_of_states["C-" + str(pair[0]) + "_" + str(pair[1])] = []
	
	for trace in event_log:
		average_trace_length += len(trace)
		start_state = None
		end_state = None
		for idx,event in enumerate(trace):
			state_transition = event["concept:name"]
			
			if start_state == None:
				start_state = state_transition.split("-")[1].split("_")[0]
				end_state = state_transition.split("-")[1].split("_")[1]
				initial_state_elapsed_seconds = decode_time(event["time:timestamp"])
				times_of_initial_state[start_state].append(initial_state_elapsed_seconds)
			else:

				time_of_state_transition = decode_time(event["time:timestamp"])
				time_of_previous_state_transition = decode_time(trace[idx-1]["time:timestamp"])
				if synthetic_data == 0:
					times_of_states["J2-" + start_state + "_" + end_state].append(time_of_state_transition - time_of_previous_state_transition)
				else:
					times_of_states["C-" + start_state + "_" + end_state].append(time_of_state_transition - time_of_previous_state_transition)
				start_state = state_transition.split("-")[1].split("_")[0]
				end_state = state_transition.split("-")[1].split("_")[1]

	for initial_state in times_of_initial_state:
		if len(times_of_initial_state[initial_state]) == 0:
			times_of_initial_state[initial_state].append(1)

	average_trace_length = int(average_trace_length/len(event_log))

	return times_of_initial_state, times_of_states, average_trace_length
	
def decode_time(timestamp):

	seconds = timestamp.second
	minutes = timestamp.minute
	hours = timestamp.hour


	return seconds + (minutes*60) + (hours*60*60)

def write_petri_net(petri_net):
	pnml_exporter.apply(petri_net["network"], petri_net["initial_marking"], output_petrinet_dir + "PN.pnml", final_marking = petri_net["final_marking"])
	
	return None
	
def write_statistics(times_of_initial_state, times_of_states, average_trace_length):

	times_of_initial_state = {int(k): v for k, v in times_of_initial_state.items()}
	times_of_initial_state = dict(sorted(times_of_initial_state.items()))
	times_of_initial_state = {str(k): v for k, v in times_of_initial_state.items()}
	times_of_states = dict(sorted(times_of_states.items(), key=lambda item: tuple(int(x) for x in item[0].split('-')[1].split('_'))))

	statistics_file = open(output_petrinet_dir + "statistics.txt", "w")

	statistics_file.write("Initial state times:\n")
	for idx,state in enumerate(times_of_initial_state):
		statistics_file.write("\t" + state + ": " + str(times_of_initial_state[state]) + "\n")
			
	statistics_file.write("Times of states:\n")		
	for idx,state_transition in enumerate(times_of_states):
		statistics_file.write("\t" + state_transition + ": " + str(times_of_states[state_transition]) + "\n")
		
	statistics_file.write("Average trace length: " + str(average_trace_length)) 	

	return None
	
def write_process_discovery_timing(process_discovery_time):

	file = open(output_timing_dir + "pd_time.txt", "w")
	file.write(str(process_discovery_time))
	file.close()
	
	return None
	
try:
	variant = sys.argv[1]
	synthetic_data = int(sys.argv[2])
except:
	print("Enter the right number of input arguments.")
	sys.exit()
	
	
event_log, centroids = read_data()
process_discovery_time = time.time()
petri_net = process_discovery(event_log, variant)
times_of_initial_state, times_of_states, average_trace_length = get_statistics(event_log, centroids, synthetic_data)
process_discovery_time = time.time() - process_discovery_time
write_petri_net(petri_net)
write_statistics(times_of_initial_state, times_of_states, average_trace_length)
write_process_discovery_timing(process_discovery_time)