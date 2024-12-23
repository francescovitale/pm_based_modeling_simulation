from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer

import os
import sys

import pandas as pd
import numpy as np
import math
import time

import pm4py

import func_timeout

input_dir = "Input/"
input_eventlogs_dir = input_dir + "EventLogs/"
input_petrinets_dir = input_dir + "PetriNets/"
input_simulatedtimeseries_dir = input_dir + "SimulatedTimeSeries/"
input_timeseries_dir = input_dir + "TimeSeries/"

output_dir = "Output/"
output_metrics_dir = output_dir + "Metrics/"
output_timeseriespairs_dir = output_dir + "TimeSeriesPairs/"
output_timing_dir = output_dir + "Timing/"

def read_event_logs(supervision_mode):

	event_logs = {}

	if supervision_mode == 0:
		for event_log in os.listdir(input_eventlogs_dir):
			n_clusters = event_log.split(".xes")[0].split("_")[0]
			event_logs[n_clusters] =  xes_importer.apply(input_eventlogs_dir + event_log)

	elif supervision_mode == 1:
		event_logs = xes_importer.apply(input_eventlogs_dir + "EL.xes")

	return event_logs

def read_petri_nets(supervision_mode):

	petri_nets = {}

	if supervision_mode == 0:

		for petri_net in os.listdir(input_petrinets_dir):
			n_clusters = petri_net.split(".pnml")[0].split("_")[0]
			pd_algorithm = petri_net.split(".pnml")[0].split("_")[1]

			if n_clusters not in list(petri_nets.keys()):
				petri_nets[n_clusters] = {}

			petri_nets[n_clusters][pd_algorithm] = {}
			petri_nets[n_clusters][pd_algorithm]["network"], petri_nets[n_clusters][pd_algorithm]["initial_marking"], petri_nets[n_clusters][pd_algorithm]["final_marking"] = pnml_importer.apply(input_petrinets_dir + petri_net)

	elif supervision_mode == 1:
		for petri_net in os.listdir(input_petrinets_dir):
			pd_algorithm = petri_net.split(".pnml")[0]
			petri_nets[pd_algorithm] = {}
			petri_nets[pd_algorithm]["network"], petri_nets[pd_algorithm]["initial_marking"], petri_nets[pd_algorithm]["final_marking"] = pnml_importer.apply(input_petrinets_dir + petri_net)

	

	return petri_nets

def read_simulated_time_series(supervision_mode):

	simulated_time_series = {}

	if supervision_mode == 0:
		for pd_configuration in os.listdir(input_simulatedtimeseries_dir):
			n_clusters = pd_configuration.split("_")[0]
			pd_algorithm = pd_configuration.split("_")[1]
			if n_clusters not in (simulated_time_series.keys()):
				simulated_time_series[n_clusters] = {}
			simulated_time_series[n_clusters][pd_algorithm] = []
			for ts in os.listdir(input_simulatedtimeseries_dir + pd_configuration):
				simulated_time_series[n_clusters][pd_algorithm].append(pd.read_csv(input_simulatedtimeseries_dir + pd_configuration + "/" + ts))

	elif supervision_mode == 1:
		for pd_configuration in os.listdir(input_simulatedtimeseries_dir):
			pd_algorithm = pd_configuration
			simulated_time_series[pd_algorithm] = []
			for ts in os.listdir(input_simulatedtimeseries_dir + pd_configuration):
				simulated_time_series[pd_algorithm].append(pd.read_csv(input_simulatedtimeseries_dir + pd_configuration + "/" + ts))

	return simulated_time_series

def read_time_series(supervision_mode):

	time_series = {}

	if supervision_mode == 0:
		for n_clusters in os.listdir(input_timeseries_dir):
			time_series[n_clusters] = []
			for ts in os.listdir(input_timeseries_dir + n_clusters):
				try:
					time_series[n_clusters].append(pd.read_csv(input_timeseries_dir + n_clusters + "/" + ts).drop(columns="C2", axis=1))
				except:
					time_series[n_clusters].append(pd.read_csv(input_timeseries_dir + n_clusters + "/" + ts).drop(columns="C", axis=1))
	elif supervision_mode == 1:
		time_series = []
		for ts in os.listdir(input_timeseries_dir):
			try:
				time_series.append(pd.read_csv(input_timeseries_dir + "/" + ts).drop(columns="C2", axis=1))
			except:
				time_series.append(pd.read_csv(input_timeseries_dir + "/" + ts).drop(columns="C", axis=1))
		


	return time_series

def compute_metrics(event_logs, petri_nets, supervision_mode):

	metrics = {}

	if supervision_mode == 0:
		for n_clusters in petri_nets:
			metrics[n_clusters] = {}
			for pd_algorithm in petri_nets[n_clusters]:
				metrics[n_clusters][pd_algorithm] = {}
				try:
					cc_timing = time.time()
					metrics[n_clusters][pd_algorithm]["fitness"], metrics[n_clusters][pd_algorithm]["precision"] = func_timeout.func_timeout(timeout=300, func=compute_fitness_precision, args=[event_logs[n_clusters], petri_nets[n_clusters][pd_algorithm]])
					metrics[n_clusters][pd_algorithm]["timing"] = time.time() - cc_timing
				except:
					metrics[n_clusters][pd_algorithm]["fitness"] = 0
					metrics[n_clusters][pd_algorithm]["precision"] = 0
					metrics[n_clusters][pd_algorithm]["timing"] = -1

	elif supervision_mode == 1:
		for pd_algorithm in petri_nets:
			metrics[pd_algorithm] = {}
			try:
				cc_timing = time.time()
				metrics[pd_algorithm]["fitness"], metrics[pd_algorithm]["precision"] = func_timeout.func_timeout(timeout=300, func=compute_fitness_precision, args=[event_logs, petri_nets[pd_algorithm]])
				metrics[pd_algorithm]["timing"] = time.time() - cc_timing
			except:
				metrics[pd_algorithm]["fitness"] = 0
				metrics[pd_algorithm]["precision"] = 0
				metrics[pd_algorithm]["timing"] = -1

	return metrics
	
def compute_fitness_precision(event_log, petri_net):
	fitness = 0
	precision = 0

	fitness = pm4py.fitness_alignments(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"])["log_fitness"]
	precision = pm4py.precision_alignments(event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"])
				
	return fitness, precision
	
	

def compute_best_time_series_pairs(simulated_time_series, time_series, metrics, synthetic_data, supervision_mode):

	mean_rmse = None
	best_time_series_pairs = {}

	if supervision_mode == 0:
		for n_clusters in simulated_time_series:
			best_time_series_pairs[n_clusters] = {}
			for pd_algorithm in simulated_time_series[n_clusters]:
				best_time_series_pairs[n_clusters][pd_algorithm] = None
				time_series_pairs = []
				for real_ts in time_series[n_clusters]:
					for simulated_ts in simulated_time_series[n_clusters][pd_algorithm]:
						if len(real_ts) > len(simulated_ts):
							last_simulated_ts_row = list(simulated_ts.iloc[-1])
							simulated_ts = pd.concat([simulated_ts,pd.DataFrame(columns=list(simulated_ts.columns),data=[last_simulated_ts_row]*(len(real_ts)-len(simulated_ts)))], ignore_index=True)
							real_ts_np = real_ts.to_numpy()
							simulated_ts_np = simulated_ts.to_numpy()
						elif len(real_ts) < len(simulated_ts):
							real_ts_np = real_ts.to_numpy()
							simulated_ts_np = simulated_ts[:-(len(simulated_ts) - len(real_ts))].to_numpy()
						else:
							real_ts_np = real_ts.to_numpy()
							simulated_ts_np = simulated_ts.to_numpy()
						
						
						rmse = np.sqrt(np.mean((real_ts_np-simulated_ts_np)**2))
						time_series_pairs.append([real_ts_np.copy(), simulated_ts_np.copy(), rmse.copy()])
				rmse_values = []
				max_rmse = float("-inf")
				min_rmse = float("inf")
				for elem in time_series_pairs:
					rmse_values.append(elem[2])
					if elem[2] < min_rmse:
						min_rmse = elem[2]
						best_time_series_pairs[n_clusters][pd_algorithm] = elem
					if elem[2] > max_rmse:
						max_rmse = elem[2]
				if synthetic_data == 1:
					best_time_series_pairs[n_clusters][pd_algorithm][0] = pd.DataFrame(columns=["C1", "C2", "C3"], data = best_time_series_pairs[n_clusters][pd_algorithm][0])
					best_time_series_pairs[n_clusters][pd_algorithm][1] = pd.DataFrame(columns=["C1", "C2", "C3"], data = best_time_series_pairs[n_clusters][pd_algorithm][1])
				elif synthetic_data == 0:
					best_time_series_pairs[n_clusters][pd_algorithm][0] = pd.DataFrame(columns=["J2XA", "J2YA", "J2ZA"], data = best_time_series_pairs[n_clusters][pd_algorithm][0])
					best_time_series_pairs[n_clusters][pd_algorithm][1] = pd.DataFrame(columns=["J2XA", "J2YA", "J2ZA"], data = best_time_series_pairs[n_clusters][pd_algorithm][1])


				mean_rmse = sum(rmse_values)/len(rmse_values)
				std_rmse = [x - mean_rmse for x in rmse_values]
				std_rmse = [x ** 2 for x in std_rmse]
				std_rmse = math.sqrt(sum(std_rmse)/(len(rmse_values)-1))

				metrics[n_clusters][pd_algorithm]["std_RMSE"] = std_rmse
				metrics[n_clusters][pd_algorithm]["mean_RMSE"] = mean_rmse
				metrics[n_clusters][pd_algorithm]["min_RMSE"] = min_rmse
				metrics[n_clusters][pd_algorithm]["max_RMSE"] = max_rmse

	elif supervision_mode == 1:
		best_time_series_pairs = {}
		for pd_algorithm in simulated_time_series:
			best_time_series_pairs[pd_algorithm] = None
			time_series_pairs = []
			for real_ts in time_series:
				for simulated_ts in simulated_time_series[pd_algorithm]:
					if len(real_ts) > len(simulated_ts):
						last_simulated_ts_row = list(simulated_ts.iloc[-1])
						simulated_ts = pd.concat([simulated_ts,pd.DataFrame(columns=list(simulated_ts.columns),data=[last_simulated_ts_row]*(len(real_ts)-len(simulated_ts)))], ignore_index=True)
						real_ts_np = real_ts.to_numpy()
						simulated_ts_np = simulated_ts.to_numpy()
					elif len(real_ts) < len(simulated_ts):
						real_ts_np = real_ts.to_numpy()
						simulated_ts_np = simulated_ts[:-(len(simulated_ts) - len(real_ts))].to_numpy()
					else:
						real_ts_np = real_ts.to_numpy()
						simulated_ts_np = simulated_ts.to_numpy()
					rmse = np.sqrt(np.mean((real_ts_np-simulated_ts_np)**2))
					time_series_pairs.append([real_ts_np.copy(), simulated_ts_np.copy(), rmse.copy()])

			rmse_values = []
			max_rmse = float("-inf")
			min_rmse = float("inf")
			for elem in time_series_pairs:
				rmse_values.append(elem[2])
				if elem[2] < min_rmse:
					min_rmse = elem[2]
					best_time_series_pairs[pd_algorithm] = elem
				if elem[2] > max_rmse:
					max_rmse = elem[2]
				if synthetic_data == 1:
					best_time_series_pairs[pd_algorithm][0] = pd.DataFrame(columns=["C1", "C2", "C3"], data = best_time_series_pairs[pd_algorithm][0])
					best_time_series_pairs[pd_algorithm][1] = pd.DataFrame(columns=["C1", "C2", "C3"], data = best_time_series_pairs[pd_algorithm][1])
				elif synthetic_data == 0:
					best_time_series_pairs[pd_algorithm][0] = pd.DataFrame(columns=["J2XA", "J2YA", "J2ZA"], data = best_time_series_pairs[pd_algorithm][0])
					best_time_series_pairs[pd_algorithm][1] = pd.DataFrame(columns=["J2XA", "J2YA", "J2ZA"], data = best_time_series_pairs[pd_algorithm][1])


			mean_rmse = sum(rmse_values)/len(rmse_values)
			std_rmse = [x - mean_rmse for x in rmse_values]
			std_rmse = [x ** 2 for x in std_rmse]
			std_rmse = math.sqrt(sum(std_rmse)/(len(rmse_values)-1))

			metrics[pd_algorithm]["std_RMSE"] = std_rmse
			metrics[pd_algorithm]["mean_RMSE"] = mean_rmse
			metrics[pd_algorithm]["min_RMSE"] = min_rmse
			metrics[pd_algorithm]["max_RMSE"] = max_rmse
		

	return best_time_series_pairs, metrics

def write_metrics(metrics, supervision_mode):

	if supervision_mode == 0:
		for n_clusters in metrics:
			for pd_algorithm in metrics[n_clusters]:
				file = open(output_metrics_dir + n_clusters + "_" + pd_algorithm + "_effectiveness.txt","w")
				file.write("fitness: " + str(metrics[n_clusters][pd_algorithm]["fitness"]) + "\n")
				file.write("precision: " + str(metrics[n_clusters][pd_algorithm]["precision"]) + "\n")
				file.write("mean_RMSE: " + str(metrics[n_clusters][pd_algorithm]["mean_RMSE"]) + "\n")
				file.write("std_RMSE: " + str(metrics[n_clusters][pd_algorithm]["std_RMSE"]) + "\n")
				file.write("min_RMSE: " + str(metrics[n_clusters][pd_algorithm]["min_RMSE"]) + "\n")
				file.write("max_RMSE: " + str(metrics[n_clusters][pd_algorithm]["max_RMSE"]))
				file.close()
				
				file = open(output_metrics_dir + n_clusters + "_" + pd_algorithm + "_cc_time.txt", "w")
				file.write(str(metrics[n_clusters][pd_algorithm]["timing"]))
				file.close()
	elif supervision_mode == 1:
		for pd_algorithm in metrics:
			file = open(output_metrics_dir + pd_algorithm + "_effectiveness.txt","w")
			file.write("fitness: " + str(metrics[pd_algorithm]["fitness"]) + "\n")
			file.write("precision: " + str(metrics[pd_algorithm]["precision"]) + "\n")
			file.write("mean_RMSE: " + str(metrics[pd_algorithm]["mean_RMSE"]) + "\n")
			file.write("std_RMSE: " + str(metrics[pd_algorithm]["std_RMSE"]) + "\n")
			file.write("min_RMSE: " + str(metrics[pd_algorithm]["min_RMSE"]) + "\n")
			file.write("max_RMSE: " + str(metrics[pd_algorithm]["max_RMSE"]))
			file.close()

			file = open(output_metrics_dir + n_clusters + "_" + pd_algorithm + "_cc_time.txt", "w")
			file.write(str(metrics[pd_algorithm]["timing"]))
			file.close()
	return None

def write_best_time_series_pairs(best_time_series_pairs, supervision_mode):

	if supervision_mode == 0:
		for n_clusters in best_time_series_pairs:
			for pd_algorithm in best_time_series_pairs[n_clusters]:
				isExist = os.path.exists(output_timeseriespairs_dir + n_clusters + "_" + pd_algorithm)
				if isExist != True:
					os.mkdir(output_timeseriespairs_dir + n_clusters + "_" + pd_algorithm)
				best_time_series_pairs[n_clusters][pd_algorithm][0].to_csv(output_timeseriespairs_dir + n_clusters + "_" + pd_algorithm + "/R_TS.csv", index=False)
				best_time_series_pairs[n_clusters][pd_algorithm][1].to_csv(output_timeseriespairs_dir + n_clusters + "_" + pd_algorithm + "/S_TS.csv", index=False)
	elif supervision_mode == 1:
		for pd_algorithm in best_time_series_pairs:
			isExist = os.path.exists(output_timeseriespairs_dir + pd_algorithm)
			if isExist != True:
				os.mkdir(output_timeseriespairs_dir + pd_algorithm)
			best_time_series_pairs[pd_algorithm][0].to_csv(output_timeseriespairs_dir + pd_algorithm + "/R_TS.csv", index=False)
			best_time_series_pairs[pd_algorithm][1].to_csv(output_timeseriespairs_dir + pd_algorithm + "/S_TS.csv", index=False)



	return None

try:
	synthetic_data = int(sys.argv[1])
	supervision_mode = int(sys.argv[2])
except:
	print("Enter the right number of input arguments.")
	sys.exit()

event_logs = read_event_logs(supervision_mode)
petri_nets = read_petri_nets(supervision_mode)
simulated_time_series = read_simulated_time_series(supervision_mode)
time_series = read_time_series(supervision_mode)
metrics = compute_metrics(event_logs, petri_nets, supervision_mode)
best_time_series_pairs, metrics = compute_best_time_series_pairs(simulated_time_series, time_series, metrics, synthetic_data, supervision_mode)
write_metrics(metrics, supervision_mode)
write_best_time_series_pairs(best_time_series_pairs, supervision_mode)



