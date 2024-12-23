from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
import pm4py

import sys
import os

import numpy as np
from itertools import combinations,permutations
import datetime

import random
import pandas as pd

import func_timeout
import time

input_dir = "Input/S/"
input_clustering_dir = input_dir + "Clustering/"
input_eventlog_dir = input_dir + "EventLog/"
input_petrinet_dir = input_dir + "PetriNet/"

output_dir = "Output/S/"
output_data_dir = output_dir + "Data/"
output_timing_dir = output_dir + "Timing/"


def read_event_log():

	event_log = xes_importer.apply(input_eventlog_dir + "EL.xes")

	return event_log

def read_petri_net():

	petri_net = {}

	petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pnml_importer.apply(input_petrinet_dir + "PN.pnml")

	return petri_net

def read_centroids():
	centroids = {}

	file = open(input_clustering_dir + "clustering_parameters.txt","r")
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

def simulate_traces(petri_net, average_trace_length):

	simulated_event_log = simulator.apply(petri_net["network"], petri_net["initial_marking"], variant=simulator.Variants.BASIC_PLAYOUT, parameters={simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 1000})
	traces = []
	for trace in simulated_event_log:
		state_transitions = []
		for event in trace:
			state_transitions.append(event["concept:name"])
		traces.append(state_transitions)
			
	return traces

def filter_traces(traces, average_trace_length, apply_constraints):

	filtered_traces = []
	
	for idx_t,trace in enumerate(traces):
		temp = []
		current_state = None
		for idx_st,state_transition in enumerate(trace):
			if current_state == None:
				current_state = state_transition.split("-")[1].split("_")[1]
				temp.append(state_transition)
			else:
				if current_state != state_transition.split("-")[1].split("_")[0]:
					continue
				else:
					current_state = state_transition.split("-")[1].split("_")[1]
					temp.append(state_transition)
		if apply_constraints == 1:
			if len(temp) >= average_trace_length - 10 and len(temp) <= average_trace_length + 10 and len(temp) > 0:
				filtered_traces.append(temp.copy())
		else:
			if len(temp) > 0:
				filtered_traces.append(temp.copy())
	return filtered_traces

def simulate_time_series(filtered_traces, centroids, times_of_initial_state, times_of_states, synthetic_data):
	simulated_time_series = []
	for trace in filtered_traces:
		rows = []
		for idx,event in enumerate(trace):
			if idx == 0:
				time_of_initial_state = random.choice(times_of_initial_state[event.split("-")[1].split("_")[0]])
				for i in range(0,time_of_initial_state):
					rows.append(centroids[event.split("-")[1].split("_")[0]])
			elif idx != 0:
				try:
					previous_state_time = random.choice(times_of_states[event])
				except:
					previous_state_time = 12
				for i in range(0,previous_state_time):
					rows.append(centroids[event.split("-")[1].split("_")[0]])
				
		if synthetic_data == 0:
			time_series = pd.DataFrame(columns = ["J2XA", "J2YA", "J2ZA"], data=rows)
		else:
			time_series = pd.DataFrame(columns = ["C1", "C2", "C3"], data=rows)
		simulated_time_series.append(time_series)

	return simulated_time_series

def generate_traces(petri_net, average_trace_length, apply_constraints):

	filtered_traces = []
	
	while len(filtered_traces) != n_traces:
		traces = simulate_traces(petri_net, average_trace_length)
		filtered_traces = filtered_traces + filter_traces(traces, average_trace_length, apply_constraints)
		if len(filtered_traces) > n_traces:
			filtered_traces = filtered_traces[0:n_traces]		
	

	return filtered_traces

def write_time_series(simulated_time_series):
	
	for idx,time_series in enumerate(simulated_time_series):
		time_series.to_csv(output_data_dir + "S_WNDW_" + str(idx) + ".csv", index=False)

	return None
	
def write_simulation_timing(simulation_time):

	file = open(output_timing_dir + "simulation_time.txt", "w")
	file.write(str(simulation_time))
	file.close()
		

try:
	n_traces = int(sys.argv[1])
	synthetic_data = int(sys.argv[2])
	simulation_technique = sys.argv[3]
except:
	print("Enter the right number of input arguments")
	sys.exit()

if simulation_technique == "process_mining":
	event_log = read_event_log()
	petri_net = read_petri_net()
	centroids = read_centroids()
	times_of_initial_state, times_of_states, average_trace_length = get_statistics(event_log, centroids, synthetic_data)

	try:
		simulation_time = time.time()
		apply_constraints = 1
		filtered_traces = func_timeout.func_timeout(timeout=60, func=generate_traces, args=[petri_net, average_trace_length, apply_constraints])
		simulation_time = time.time() - simulation_time
	except func_timeout.FunctionTimedOut:
		simulation_time = time.time()
		apply_constraints = 0
		filtered_traces = func_timeout.func_timeout(timeout=180, func=generate_traces, args=[petri_net, average_trace_length, apply_constraints])
		simulation_time = time.time() - simulation_time
	
	simulated_time_series = simulate_time_series(filtered_traces, centroids, times_of_initial_state, times_of_states, synthetic_data)
	write_time_series(simulated_time_series)
	write_simulation_timing(simulation_time)


