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
input_petrinet_dir = input_dir + "PetriNet/"
input_clustering_dir = input_dir + "Clustering/"

output_dir = "Output/S/"
output_data_dir = output_dir + "Data/"
output_timing_dir = output_dir + "Timing/"
	
def read_statistics():
    times_of_initial_state = {}
    times_of_states = {}
    average_trace_length = None

    with open(input_petrinet_dir + "statistics.txt", "r") as f:
        lines = f.readlines()

    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Detect sections
        if line.startswith("Initial state times:"):
            current_section = "initial"
            continue
        elif line.startswith("Times of states:"):
            current_section = "states"
            continue
        elif line.startswith("Average trace length:"):
            average_trace_length = float(line.split(":")[1].strip())
            continue

        # Parse the dictionary entries
        if current_section == "initial":
            key, value = line.split(":", 1)
            # Convert key to string (character)
            key_str = key.strip()
            values_list = [int(x.strip()) for x in value.strip()[1:-1].split(",") if x.strip()]
            times_of_initial_state[key_str] = values_list
        elif current_section == "states":
            key, value = line.split(":", 1)
            value = value.strip()
            if value == "[]":
                values_list = []
            else:
                values_list = [int(x.strip()) for x in value[1:-1].split(",") if x.strip()]
            times_of_states[key.strip()] = values_list

    return times_of_initial_state, times_of_states, average_trace_length	

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

def read_petri_net():

	petri_net = {}

	petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pnml_importer.apply(input_petrinet_dir + "PN.pnml")

	return petri_net

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

def simulate_time_series(simulated_traces, centroids, times_of_initial_state, times_of_states, synthetic_data):
	simulated_time_series = []
	simulated_time_series_state_times = []
	for trace in simulated_traces:
		state_times = []
		rows = []
		
		for idx,event in enumerate(trace):
			if idx == 0:
				#time_of_initial_state = random.choices(times_of_initial_state[event.split("-")[1].split("_")[0]], weights=times_of_initial_state[event.split("-")[1].split("_")[0]], k=1)[0]
				time_of_initial_state = random.choice(times_of_initial_state[event.split("-")[1].split("_")[0]])
				for i in range(0,time_of_initial_state):
					rows.append(centroids[event.split("-")[1].split("_")[0]])
				state_times.append(time_of_initial_state)	
			elif idx != 0:
				try:
					previous_state_time = random.choice(times_of_states[event])
				except:
					previous_state_time = 1
				for i in range(0,previous_state_time):
					rows.append(centroids[event.split("-")[1].split("_")[0]])
				state_times.append(previous_state_time)
		if synthetic_data == 0:
			time_series = pd.DataFrame(columns = ["J2XA", "J2YA", "J2ZA"], data=rows)
		else:
			time_series = pd.DataFrame(columns = ["C1", "C2", "C3"], data=rows)
			
		simulated_time_series.append(time_series)
		simulated_time_series_state_times.append(state_times)

	return simulated_time_series, simulated_time_series_state_times

def generate_traces(petri_net, average_trace_length, apply_constraints):

	filtered_traces = []
	
	while len(filtered_traces) != n_traces:
		traces = simulate_traces(petri_net, average_trace_length)
		filtered_traces = filtered_traces + filter_traces(traces, average_trace_length, apply_constraints)
		if len(filtered_traces) > n_traces:
			filtered_traces = filtered_traces[0:n_traces]		
	

	return filtered_traces

def write_time_series(simulated_time_series, simulated_traces, simulated_time_series_state_times):
	
	for idx,time_series in enumerate(simulated_time_series):
		isExist = os.path.exists(output_data_dir + "S_" + str(idx))
		if not isExist:
			os.makedirs(output_data_dir + "S_" + str(idx))
		time_series.to_csv(output_data_dir + "S_" + str(idx) + "/WNDW_" + str(idx) + ".csv", index=False)
		timings = open(output_data_dir + "S_" + str(idx) + "/timings.txt", "w")
		for idx_e, event in enumerate(simulated_traces[idx]):
			if idx_e < len(simulated_traces[idx]) - 1:
				timings.write("State transition: " + event + ", time: " + str(simulated_time_series_state_times[idx][idx_e]) + "\n")
			else:
				timings.write("State transition: " + event + ", time: " + str(simulated_time_series_state_times[idx][idx_e]))

		

	return None
	
def write_simulation_timing(simulation_time):

	file = open(output_timing_dir + "simulation_time.txt", "w")
	file.write(str(simulation_time))
	file.close()
		

try:
	n_traces = int(sys.argv[1])
	synthetic_data = int(sys.argv[2])
except:
	print("Enter the right number of input arguments")
	sys.exit()

petri_net = read_petri_net()
centroids = read_centroids()
times_of_initial_state, times_of_states, average_trace_length = read_statistics()

try:
	simulation_time = time.time()
	apply_constraints = 1
	simulated_traces = func_timeout.func_timeout(timeout=60, func=generate_traces, args=[petri_net, average_trace_length, apply_constraints])
	simulation_time = time.time() - simulation_time
except func_timeout.FunctionTimedOut:
	simulation_time = time.time()
	apply_constraints = 0
	simulated_traces = func_timeout.func_timeout(timeout=180, func=generate_traces, args=[petri_net, average_trace_length, apply_constraints])
	simulation_time = time.time() - simulation_time
	
simulated_time_series, simulated_time_series_state_times = simulate_time_series(simulated_traces, centroids, times_of_initial_state, times_of_states, synthetic_data)
write_time_series(simulated_time_series, simulated_traces, simulated_time_series_state_times)
write_simulation_timing(simulation_time)


