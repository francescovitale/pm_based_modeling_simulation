from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr

import os
import sys

import pandas as pd
import numpy as np
import math
import time

import pm4py

import func_timeout


input_dir = "Input/"

output_dir = "Output/"
output_metrics_dir = output_dir + "Metrics/"
output_timeseriespairs_dir = output_dir + "TimeSeriesPairs/"
output_timing_dir = output_dir + "Timing/"

def read_event_logs():

	event_logs = {}

	for rep_number in os.listdir(input_dir):
		event_logs[rep_number] = {}
		for result in os.listdir(input_dir + rep_number):
			anomaly_type = result.split("_")[0]
			n_clusters = result.split("_")[1]
			algorithm = result.split("_")[2]
		
			if anomaly_type not in event_logs[rep_number].keys():
				event_logs[rep_number][anomaly_type] = {}
			if n_clusters not in event_logs[rep_number][anomaly_type].keys():
				event_logs[rep_number][anomaly_type][n_clusters] = {}
			event_logs[rep_number][anomaly_type][n_clusters][algorithm] = xes_importer.apply(input_dir + rep_number + "/" + result + "/TestEventLog/EL_TST.xes")

	return event_logs

def read_petri_nets():

	petri_nets = {}

	for rep_number in os.listdir(input_dir):
		petri_nets[rep_number] = {}
		for result in os.listdir(input_dir + rep_number):
			anomaly_type = result.split("_")[0]
			n_clusters = result.split("_")[1]
			algorithm = result.split("_")[2]
		
			if anomaly_type not in petri_nets[rep_number].keys():
				petri_nets[rep_number][anomaly_type] = {}
			if n_clusters not in petri_nets[rep_number][anomaly_type].keys():
				petri_nets[rep_number][anomaly_type][n_clusters] = {}
			petri_nets[rep_number][anomaly_type][n_clusters][algorithm] = {}
			petri_nets[rep_number][anomaly_type][n_clusters][algorithm]["structure"] = {}
			
			temp = {}
			temp["network"], temp["initial_marking"], temp["final_marking"] = pnml_importer.apply(input_dir + rep_number + "/" + result + "/PetriNet/PN.pnml")
			petri_nets[rep_number][anomaly_type][n_clusters][algorithm]["structure"] = temp
			petri_nets[rep_number][anomaly_type][n_clusters][algorithm]["statistics"] = read_statistics(input_dir + rep_number + "/" + result + "/PetriNet/statistics.txt")

	return petri_nets
	
def read_statistics(statistics_dir):
	times_of_initial_state = {}
	times_of_states = {}
	average_trace_length = None

	with open(statistics_dir, "r") as f:
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

	return times_of_states	
	
def read_simulated_time_series():

	simulated_time_series = {}
	
	for rep_number in os.listdir(input_dir):
		simulated_time_series[rep_number] = {}
		for result in os.listdir(input_dir + rep_number):
			anomaly_type = result.split("_")[0]
			n_clusters = result.split("_")[1]
			algorithm = result.split("_")[2]
		
			if anomaly_type not in simulated_time_series[rep_number].keys():
				simulated_time_series[rep_number][anomaly_type] = {}
			if n_clusters not in simulated_time_series[rep_number][anomaly_type].keys():
				simulated_time_series[rep_number][anomaly_type][n_clusters] = {}
			simulated_time_series[rep_number][anomaly_type][n_clusters][algorithm] = []
			
			for ts in os.listdir(input_dir + rep_number + "/" + result + "/SimulatedTimeSeries"):
				ts_number = ts.split(".csv")[0].split("_")[-1]
				temp = []
				temp.append(pd.read_csv(input_dir + rep_number + "/" + result + "/SimulatedTimeSeries/" + ts + "/WNDW_" + ts_number + ".csv"))
				temp.append(read_state_transitions(input_dir + rep_number + "/" + result + "/SimulatedTimeSeries/" + ts + "/timings.txt"))
				simulated_time_series[rep_number][anomaly_type][n_clusters][algorithm].append(temp)

	return simulated_time_series

def read_state_transitions(transitions_info_dir):
	transitions = []
	
	with open(transitions_info_dir, 'r') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue  # skip empty lines
			
			try:
				# Split into parts
				prefix, time_part = line.split(", time:")
				transition = prefix.split("State transition:")[1].strip()
				time_value = int(time_part.strip())
				
				transitions.append({
					"transition": transition,
					"time": time_value
				})
			except Exception as e:
				print(f"Skipping malformed line: {line} ({e})")
	
	return transitions

def read_time_series():

	time_series = {}
	
	for rep_number in os.listdir(input_dir):
		time_series[rep_number] = {}
		for result in os.listdir(input_dir + rep_number):
			anomaly_type = result.split("_")[0]
			n_clusters = result.split("_")[1]
			algorithm = result.split("_")[2]
		
			if anomaly_type not in time_series[rep_number].keys():
				time_series[rep_number][anomaly_type] = {}
			if n_clusters not in time_series[rep_number][anomaly_type].keys():
				time_series[rep_number][anomaly_type][n_clusters] = {}
			time_series[rep_number][anomaly_type][n_clusters][algorithm] = []
			
			for ts in os.listdir(input_dir + rep_number + "/" + result + "/TimeSeries"):
				time_series[rep_number][anomaly_type][n_clusters][algorithm].append(pd.read_csv(input_dir + rep_number + "/" + result + "/TimeSeries/" + ts).drop("C2", axis=1))
				
	return time_series
	
def compute_fitness_simplicity(event_log, petri_net):
	
	#f1 = 0.0
	
	fitness = 0
	precision = 0

	try:
		fitness = func_timeout.func_timeout(timeout=60, func=pm4py.fitness_alignments, args=[event_log, petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"]])
		fitness = fitness["log_fitness"]
	except:
		fitness = np.nan
		
	simplicity_arc = pm4py.simplicity_petri_net(petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"], variant='arc_degree')
	
	try:
		simplicity_cycle = func_timeout.func_timeout(timeout=60, func=pm4py.simplicity_petri_net, args=[petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"], "extended_cyclomatic"])
	except:
		simplicity_cycle = np.nan
	
	return fitness, simplicity_arc, simplicity_cycle

def compute_distributions(statistics):

	distributions = {}

	for state_transition in statistics:
	
		values = statistics[state_transition]
		total = len(values)
		dist = {}
		for v in values:
			dist[v] = dist.get(v, 0) + 1
		distributions[state_transition] = {k: v / total for k, v in dist.items()}

	return distributions
	
def merge_distributions(dicts, weights=None):
	if weights is None:
		weights = [1.0] * len(dicts)
	if len(weights) != len(dicts):
		raise ValueError("len(weights) must match len(dicts)")

	# accumulate weighted "counts" for each outer_key and inner value
	accum = {}  # outer_key -> {value: accumulated_weighted_prob}
	all_outer_keys = set()
	for d, w in zip(dicts, weights):
		all_outer_keys.update(d.keys())
		for outer_key, inner in d.items():
			if not inner:
				# empty inner dict -> contributes nothing
				continue
			bucket = accum.setdefault(outer_key, {})
			for val, prob in inner.items():
				bucket[val] = bucket.get(val, 0.0) + float(prob) * float(w)

	# normalize each outer_key's accumulated counts into probabilities
	merged = {}
	for outer_key in all_outer_keys:
		bucket = accum.get(outer_key, {})
		total = sum(bucket.values())
		if total > 0:
			merged[outer_key] = {val: cnt / total for val, cnt in bucket.items()}
		else:
			# all inputs had empty inner dict for this outer_key
			merged[outer_key] = {}

	return merged
	
def compute_pm_metrics(event_logs, petri_nets):

	pm_metrics = {}
	
	temp_rep_number = list(event_logs.keys())[0]
	for anomaly_type in event_logs[temp_rep_number]:
		pm_metrics[anomaly_type] = {}
		for n_clusters in event_logs[temp_rep_number][anomaly_type]:
			pm_metrics[anomaly_type][n_clusters] = {}
			for algorithm in event_logs[temp_rep_number][anomaly_type][n_clusters]:
				pm_metrics[anomaly_type][n_clusters][algorithm] = {}
				
				pm_metrics[anomaly_type][n_clusters][algorithm]["F"] = []
				pm_metrics[anomaly_type][n_clusters][algorithm]["S_arc"] = []
				pm_metrics[anomaly_type][n_clusters][algorithm]["S_cycle"] = []
				pm_metrics[anomaly_type][n_clusters][algorithm]["Distributions"] = []
				
	for rep_number in event_logs:
		for anomaly_type in event_logs[rep_number]:
			for n_clusters in event_logs[rep_number][anomaly_type]:
				for algorithm in event_logs[rep_number][anomaly_type][n_clusters]:
					temp_event_log = event_logs[rep_number][anomaly_type][n_clusters][algorithm]
					temp_petri_net = petri_nets[rep_number][anomaly_type][n_clusters][algorithm]["structure"]
					temp_distributions = compute_distributions(petri_nets[rep_number][anomaly_type][n_clusters][algorithm]["statistics"])
					fitness, simplicity_arc, simplicity_cycle = compute_fitness_simplicity(temp_event_log, temp_petri_net)

					pm_metrics[anomaly_type][n_clusters][algorithm]["F"].append(fitness)
					pm_metrics[anomaly_type][n_clusters][algorithm]["S_arc"].append(simplicity_arc)
					pm_metrics[anomaly_type][n_clusters][algorithm]["S_cycle"].append(simplicity_cycle)
					pm_metrics[anomaly_type][n_clusters][algorithm]["Distributions"].append(temp_distributions)
	
	for anomaly_type in pm_metrics:
		for n_clusters in pm_metrics[anomaly_type]:
			for algorithm in pm_metrics[anomaly_type][n_clusters]:
					temp_fitness = {}
					temp_fitness["mean"], temp_fitness["std"] = compute_mean_std(pm_metrics[anomaly_type][n_clusters][algorithm]["F"])
					pm_metrics[anomaly_type][n_clusters][algorithm]["F"] = temp_fitness
					
					temp_s = {}
					temp_s["mean"], temp_s["std"] = compute_mean_std(pm_metrics[anomaly_type][n_clusters][algorithm]["S_arc"])
					pm_metrics[anomaly_type][n_clusters][algorithm]["S_arc"] = temp_s
					
					temp_s = {}
					temp_s["mean"], temp_s["std"] = compute_mean_std(pm_metrics[anomaly_type][n_clusters][algorithm]["S_cycle"])
					pm_metrics[anomaly_type][n_clusters][algorithm]["S_cycle"] = temp_s
					
					pm_metrics[anomaly_type][n_clusters][algorithm]["Distributions"] = merge_distributions(pm_metrics[anomaly_type][n_clusters][algorithm]["Distributions"])
				
	return pm_metrics

def compute_mean_std(values):
	n = len(values)
	mean = sum(values) / n
	variance = sum((x - mean) ** 2 for x in values) / (n)  # sample variance
	std = math.sqrt(variance)
	return mean, std

def compute_best_time_series_pairs(simulated_time_series, time_series, pm_metrics):

	best_time_series_pairs = {}

	temp_rep_number = list(event_logs.keys())[0]
	for anomaly_type in event_logs[temp_rep_number]:
		best_time_series_pairs[anomaly_type] = {}
		for n_clusters in event_logs[temp_rep_number][anomaly_type]:
			best_time_series_pairs[anomaly_type][n_clusters] = {}
			for algorithm in event_logs[temp_rep_number][anomaly_type][n_clusters]:
				best_time_series_pairs[anomaly_type][n_clusters][algorithm] = []

				# initialize metrics
				pm_metrics[anomaly_type][n_clusters][algorithm]["min_R2"] = []
				pm_metrics[anomaly_type][n_clusters][algorithm]["max_R2"] = []
				pm_metrics[anomaly_type][n_clusters][algorithm]["mean_R2"] = []
				pm_metrics[anomaly_type][n_clusters][algorithm]["min_RMSE"] = []
				pm_metrics[anomaly_type][n_clusters][algorithm]["max_RMSE"] = []
				pm_metrics[anomaly_type][n_clusters][algorithm]["mean_RMSE"] = []

	# Loop over repetitions and series
	for rep_number in event_logs:
		for anomaly_type in event_logs[rep_number]:
			for n_clusters in event_logs[rep_number][anomaly_type]:
				for algorithm in event_logs[rep_number][anomaly_type][n_clusters]:

					temp_time_series_pairs = []

					for simulated_ts in simulated_time_series[rep_number][anomaly_type][n_clusters][algorithm]:
						pairs = []

						for real_ts in time_series[rep_number][anomaly_type][n_clusters][algorithm]:
							real_ts_np = real_ts.to_numpy()
							simulated_ts_np = simulated_ts[0].to_numpy()

							# pad arrays to same length
							max_len = max(len(real_ts_np), len(simulated_ts_np))
							n_features = real_ts_np.shape[1] if real_ts_np.ndim > 1 else 1

							real_padded = np.zeros((max_len, n_features))
							sim_padded = np.zeros((max_len, n_features))

							if real_ts_np.ndim == 1:
								real_padded[:len(real_ts_np), 0] = real_ts_np
								sim_padded[:len(simulated_ts_np), 0] = simulated_ts_np
							else:
								real_padded[:len(real_ts_np), :] = real_ts_np
								sim_padded[:len(simulated_ts_np), :] = simulated_ts_np

							# compute R² and RMSE
							if n_features == 1:
								r2 = r2_score(real_padded.flatten(), sim_padded.flatten())
								rmse = np.sqrt(np.mean((real_padded.flatten() - sim_padded.flatten()) ** 2))
							else:
								r2 = np.mean([
									r2_score(real_padded[:, i], sim_padded[:, i]) for i in range(n_features)
								])
								rmse = np.mean([
									np.sqrt(np.mean((real_padded[:, i] - sim_padded[:, i]) ** 2)) for i in range(n_features)
								])

							# convert arrays to DataFrames
							real_df = pd.DataFrame(real_ts_np, columns=["J2XA", "J2YA", "J2ZA"])
							sim_df = pd.DataFrame(simulated_ts_np, columns=["J2XA", "J2YA", "J2ZA"])

							pairs.append({
								"R_TS": real_df,
								"S_TS": sim_df,
								"R2": r2,
								"RMSE": rmse,
								"transitions": simulated_ts[1]
							})

						# select pair with max R²
						best_pair = max(pairs, key=lambda x: x["R2"])
						temp_time_series_pairs.append(best_pair)

					# compute summary statistics
					r2_values = [p["R2"] for p in temp_time_series_pairs]
					rmse_values = [p["RMSE"] for p in temp_time_series_pairs]

					pm_metrics[anomaly_type][n_clusters][algorithm]["min_R2"].append(min(r2_values))
					pm_metrics[anomaly_type][n_clusters][algorithm]["max_R2"].append(max(r2_values))
					pm_metrics[anomaly_type][n_clusters][algorithm]["mean_R2"].append(np.mean(r2_values))

					pm_metrics[anomaly_type][n_clusters][algorithm]["min_RMSE"].append(min(rmse_values))
					pm_metrics[anomaly_type][n_clusters][algorithm]["max_RMSE"].append(max(rmse_values))
					pm_metrics[anomaly_type][n_clusters][algorithm]["mean_RMSE"].append(np.mean(rmse_values))

					# store best pair by max R²
					best_time_series_pairs[anomaly_type][n_clusters][algorithm].append(
						max(temp_time_series_pairs, key=lambda x: x["R2"])
					)

	# compute mean/std for metrics
	for anomaly_type in pm_metrics:
		for n_clusters in pm_metrics[anomaly_type]:
			for algorithm in pm_metrics[anomaly_type][n_clusters]:
				for metric in ["min_R2", "mean_R2", "max_R2", "min_RMSE", "mean_RMSE", "max_RMSE"]:
					temp = {}
					temp["mean"], temp["std"] = compute_mean_std(pm_metrics[anomaly_type][n_clusters][algorithm][metric])
					pm_metrics[anomaly_type][n_clusters][algorithm][metric] = temp

	return best_time_series_pairs, pm_metrics


def write_best_time_series_pairs(best_time_series_pairs):
	for anomaly_type in best_time_series_pairs:
		anomaly_dir = os.path.join(output_timeseriespairs_dir, anomaly_type)
		os.makedirs(anomaly_dir, exist_ok=True)

		for n_clusters in best_time_series_pairs[anomaly_type]:
			for algorithm in best_time_series_pairs[anomaly_type][n_clusters]:
				
				pair_dir = os.path.join(anomaly_dir, f"{n_clusters}_{algorithm}")
				os.makedirs(pair_dir, exist_ok=True)

				# There should be only one best pair per algorithm
				pair = best_time_series_pairs[anomaly_type][n_clusters][algorithm][0]

				# Write real and simulated time series
				pair["R_TS"].to_csv(os.path.join(pair_dir, "R_TS.csv"), index=False)
				pair["S_TS"].to_csv(os.path.join(pair_dir, "S_TS.csv"), index=False)

				# Write R², RMSE, and transitions
				with open(os.path.join(pair_dir, "transitions_metrics.txt"), "w") as f:
					f.write("Transitions and time:\n")
					for item in pair["transitions"]:
						f.write(f"\t{item['transition']}, {item['time']}\n")
					f.write(f"R²: {pair['R2']:.4f}\n")
					f.write(f"RMSE: {pair['RMSE']:.4f}\n")

	return None

def write_distribution_to_file(merged_dist, file_obj):
	for outer_key, inner_dict in merged_dist.items():
		file_obj.write(f"{outer_key}:\n")
		if inner_dict:
			for val, prob in inner_dict.items():
				file_obj.write(f"	{val}: {prob:.6f}\n")
		else:
			file_obj.write("	{}\n")

def write_metrics(pm_metrics):

	for anomaly_type in pm_metrics:
		isExist = os.path.exists(output_metrics_dir + anomaly_type)
		if not isExist:
			os.makedirs(output_metrics_dir + anomaly_type)
		for n_clusters in pm_metrics[anomaly_type]:
			for algorithm in pm_metrics[anomaly_type][n_clusters]:
				metrics_file = open(output_metrics_dir + anomaly_type + "/" + n_clusters + "_" + algorithm + ".txt", "w")
				for metric in pm_metrics[anomaly_type][n_clusters][algorithm]:
					
					if metric != "Distributions":
						mean = pm_metrics[anomaly_type][n_clusters][algorithm][metric]["mean"]
						std = pm_metrics[anomaly_type][n_clusters][algorithm][metric]["std"]
						metrics_file.write(metric + ": " + str(mean) + " +/- " + str(std) + "\n")
					elif metric == "Distributions":
						metrics_file.write("State transition distributions:")
						write_distribution_to_file(pm_metrics[anomaly_type][n_clusters][algorithm][metric], metrics_file)
						
				metrics_file.close()
			
	return None

event_logs = read_event_logs()
petri_nets = read_petri_nets()
simulated_time_series = read_simulated_time_series()
time_series = read_time_series()
pm_metrics = compute_pm_metrics(event_logs, petri_nets)
best_time_series_pairs, pm_metrics = compute_best_time_series_pairs(simulated_time_series, time_series, pm_metrics)
write_metrics(pm_metrics)
write_best_time_series_pairs(best_time_series_pairs)


