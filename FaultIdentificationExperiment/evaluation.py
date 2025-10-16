from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.importer import importer as pnml_importer
from sklearn.metrics import r2_score, mean_squared_error
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter

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

def read_training_test_set():

	'''
	the training set is structured as follows:
		- Rep number
			-- Anomaly type (V or W)
				--- Number of clusters (2 -- 12)
					---- PD algorithm (hm, ilp75, imf75)
						----- Petri net of the anomaly type with the selected number of clusters and PD algorithm
						----- Simulated time series of the anomaly type with the selected number of clusters and PD algorithm
	'''
	training_set = {}
	
	
	'''
	the test set is structured as follows:
		- Rep number
			-- Anomaly type (V or W)
				--- Number of clusters (2 -- 12)
					---- PD algorithm (hm, ilp75, imf75)
						----- V event log (positive class if anomaly type is V, negative otherwise)
						----- W event log (negative class if anomaly type is W, negative otherwise)
	'''
	test_set = {}
	
	for rep_number in os.listdir(input_dir):
		training_set[rep_number] = {}
		test_set[rep_number] = {}
		
		for configuration in os.listdir(input_dir + rep_number):
			anomaly_type = configuration.split("_")[0]
			if anomaly_type not in training_set[rep_number].keys():
				training_set[rep_number][anomaly_type] = {}
				test_set[rep_number][anomaly_type] = {}
			number_of_clusters = configuration.split("_")[1]
			if number_of_clusters not in training_set[rep_number][anomaly_type].keys():
				training_set[rep_number][anomaly_type][number_of_clusters] = {}
				test_set[rep_number][anomaly_type][number_of_clusters] = {}
			pd_algorithm = configuration.split("_")[2]
			if pd_algorithm not in training_set[rep_number][anomaly_type][number_of_clusters].keys():
				training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm] = {}
				test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm] = {}
			
			# Positive training samples
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"] = {}
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["PN"] = {}
			temp_pn = {}
			temp_pn["network"], temp_pn["initial_marking"], temp_pn["final_marking"] = pnml_importer.apply(input_dir + rep_number + "/" + configuration + "/PetriNet/PN.pnml")
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["PN"] = temp_pn
			
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["TS"] = []
			simulated_ts_dir = os.path.join(input_dir, rep_number, configuration, "SimulatedTimeSeries")
			folders = sorted(
				[f for f in os.listdir(simulated_ts_dir) if f.startswith("S_") and os.path.isdir(os.path.join(simulated_ts_dir, f))],
				key=lambda x: int(x.split('_')[1])
			)
			temp_simulated_ts = []
			for folder in folders:
				folder_path = os.path.join(simulated_ts_dir, folder)
				label = folder.split('_')[1]
				csv_path = os.path.join(folder_path, f"WNDW_{label}.csv")
				df = pd.read_csv(csv_path)
				temp_simulated_ts.append(df)
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["TS"] = temp_simulated_ts
			
			# Negative training samples
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"] = {}
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["PN"] = {}
			if anomaly_type == "V":
				neg_config = f"W_{number_of_clusters}_{pd_algorithm}"
			else:
				neg_config = f"V_{number_of_clusters}_{pd_algorithm}"
			pnml_dir = os.path.join(input_dir, rep_number, neg_config, "PetriNet", "PN.pnml")
			temp_pn = {}
			temp_pn["network"], temp_pn["initial_marking"], temp_pn["final_marking"] = pnml_importer.apply(pnml_dir)
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["PN"] = temp_pn
			
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["TS"] = []
			if anomaly_type == "V":
				neg_config = f"W_{number_of_clusters}_{pd_algorithm}"
			else:
				neg_config = f"V_{number_of_clusters}_{pd_algorithm}"
				
			simulated_ts_dir = os.path.join(input_dir, rep_number, neg_config, "SimulatedTimeSeries")
			folders = sorted(
				[f for f in os.listdir(simulated_ts_dir) if f.startswith("S_") and os.path.isdir(os.path.join(simulated_ts_dir, f))],
				key=lambda x: int(x.split('_')[1])
			)
			temp_simulated_ts = []
			for folder in folders:
				folder_path = os.path.join(simulated_ts_dir, folder)
				label = folder.split('_')[1]
				csv_path = os.path.join(folder_path, f"WNDW_{label}.csv")
				df = pd.read_csv(csv_path)
				temp_simulated_ts.append(df)
			training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["TS"] = temp_simulated_ts
			
			# Positive test samples
			test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"] = {}
			test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["TS"] = []
			ts_dir = os.path.join(input_dir, rep_number, configuration, "TimeSeries")
			files = sorted(
				[f for f in os.listdir(ts_dir) if f.startswith("WNDW_") and f.endswith(".csv")],
				key=lambda x: int(x.split('_')[1].split('.')[0])
			)
			temp_ts = [pd.read_csv(os.path.join(ts_dir, fname)).drop("C2", axis=1) for fname in files]
			test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["TS"] = temp_ts
			pos_el_path = os.path.join(input_dir, rep_number, configuration, "TestEventLog", "EL_TST.xes")
			test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["EL"] = xes_importer.apply(pos_el_path)

			# Negative samples
			test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"] = {}
			test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["TS"] = []
			if anomaly_type == "V":
				neg_config = f"W_{number_of_clusters}_{pd_algorithm}"
			else:
				neg_config = f"V_{number_of_clusters}_{pd_algorithm}"
			ts_dir = os.path.join(input_dir, rep_number, neg_config, "TimeSeries")
			files = sorted(
				[f for f in os.listdir(ts_dir) if f.startswith("WNDW_") and f.endswith(".csv")],
				key=lambda x: int(x.split('_')[1].split('.')[0])
			)
			temp_ts = [pd.read_csv(os.path.join(ts_dir, fname)).drop("C2", axis=1) for fname in files]
			test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["TS"] = temp_ts
			neg_el_path = os.path.join(input_dir, rep_number, neg_config, "TestEventLog", "EL_TST.xes")
			test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["EL"] = xes_importer.apply(neg_el_path)

	return training_set, test_set
	
def classify_windows(training_set, test_set):

	f1_score = {}
	evaluation_stats = {}
	
	for anomaly_type in training_set[list(training_set.keys())[0]]:
		f1_score[anomaly_type] = {}
		evaluation_stats[anomaly_type] = {}
		for number_of_clusters in training_set[list(training_set.keys())[0]][anomaly_type]:
			f1_score[anomaly_type][number_of_clusters] = {}
			evaluation_stats[anomaly_type][number_of_clusters] = {}
			for pd_algorithm in training_set[list(training_set.keys())[0]][anomaly_type][number_of_clusters]:
				f1_score[anomaly_type][number_of_clusters][pd_algorithm] = []
				evaluation_stats[anomaly_type][number_of_clusters][pd_algorithm] = []
	
	for rep_number in training_set:
		for anomaly_type in training_set[rep_number]:
			for number_of_clusters in training_set[rep_number][anomaly_type]:
				for pd_algorithm in training_set[rep_number][anomaly_type][number_of_clusters]:
					classifications = []
					temp_evaluation_stats = []
					if len(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["TS"]) != len(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["EL"]):
						test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["EL"] = fill_missing_cases(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["EL"])
					if len(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["TS"]) != len(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["EL"]):
						test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["EL"] = fill_missing_cases(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["EL"])

					try:
						for trace, ts in zip(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["EL"], test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["TS"]):
							elapsed = 0.0
							
							start_time = time.perf_counter()
							fitness_p = compute_fitness([trace], training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["PN"])
							fitness_n = compute_fitness([trace], training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["PN"])
							end_time = time.perf_counter()
							elapsed = end_time - start_time
							
							r2_p, rmse_p = get_best_simulated_metrics(
								ts,
								training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["TS"]
							)
							r2_n, rmse_n = get_best_simulated_metrics(
								ts,
								training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["TS"]
							)
							
							temp_evaluation_stats.append({
								"time": elapsed,
								"fitness_ratio": fitness_p / fitness_n if fitness_n != 0 else float('nan'),
								"rmse_ratio": rmse_n / rmse_p if rmse_n != 0 else float('nan'),
								"r2_ratio": robust_ratio(r2_p, r2_n)
							})
							
							votes = []
							votes.append("P" if rmse_p < rmse_n else "N")
							votes.append("P" if r2_p > r2_n else "N")
							votes.append("P" if fitness_p > fitness_n else "N")
							classified_type = 1 if votes.count("P") > votes.count("N") else 0
							classifications.append([classified_type, 1])
							
						for trace, ts in zip(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["EL"], test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["TS"]):
						
							fitness_p = compute_fitness([trace], training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["PN"])
							fitness_n = compute_fitness([trace], training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["PN"])
						
							r2_p, rmse_p = get_best_simulated_metrics(
								ts,
								training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["TS"]
							)
							r2_n, rmse_n = get_best_simulated_metrics(
								ts,
								training_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["N"]["TS"]
							)
							votes = []
							votes.append("P" if rmse_p < rmse_n else "N")
							votes.append("P" if r2_p > r2_n else "N")
							votes.append("P" if fitness_p > fitness_n else "N")
							classified_type = 1 if votes.count("P") > votes.count("N") else 0
							classifications.append([classified_type, 0])
							
					except:
						temp_evaluation_stats = []
						classifications = []
					
					evaluation_stats[anomaly_type][number_of_clusters][pd_algorithm].append(compute_mean_evaluation_stats(temp_evaluation_stats))	
					f1_score[anomaly_type][number_of_clusters][pd_algorithm].append(compute_f1(classifications))

	return f1_score, evaluation_stats

def robust_ratio(a, b):

	if a == 0 and b == 0:
		return float('nan')  # or 1, depending on your preference
	a_abs = abs(a)
	b_abs = abs(b)
	return a_abs / b_abs if a_abs >= b_abs else b_abs / a_abs
	
def fill_missing_cases(log):

	# Convert to a regular list to modify
	new_log = list(log)

	# Extract existing case IDs (numeric)
	try:
		case_ids = sorted(int(trace.attributes['concept:name']) for trace in new_log)
	except ValueError:
		raise ValueError("Case IDs must be numeric strings (e.g., '0', '1', '2').")

	# Find missing case IDs
	min_case, max_case = min(case_ids), max(case_ids)
	all_cases = set(range(min_case, max_case + 1))
	missing_cases = sorted(all_cases - set(case_ids))

	# Add empty traces for missing cases
	for cid in missing_cases:
		empty_trace = Trace()
		empty_trace.attributes['concept:name'] = str(cid)
		empty_trace.attributes['is_empty'] = True
		new_log.append(empty_trace)

	# Sort by numeric case ID
	new_log_sorted = sorted(new_log, key=lambda trace: int(trace.attributes['concept:name']))

	# Return as EventLog
	return EventLog(new_log_sorted)

def get_best_simulated_metrics(time_series, simulated_time_series):
	best_r2 = -np.inf
	best_rmse = np.inf

	# Convert the real time series to numpy
	real_ts_np = time_series.to_numpy()
	n_features = real_ts_np.shape[1] if real_ts_np.ndim > 1 else 1

	for simulated_ts in simulated_time_series:

		# Convert simulated TS to numpy
		sim_np = simulated_ts[0].to_numpy() if isinstance(simulated_ts, (list, tuple)) else simulated_ts.to_numpy()

		# Pad arrays to same length
		max_len = max(len(real_ts_np), len(sim_np))
		real_padded = np.zeros((max_len, n_features))
		sim_padded = np.zeros((max_len, n_features))

		if real_ts_np.ndim == 1:
			real_padded[:len(real_ts_np), 0] = real_ts_np
			sim_padded[:len(sim_np), 0] = sim_np
		else:
			real_padded[:len(real_ts_np), :] = real_ts_np
			sim_padded[:len(sim_np), :] = sim_np

		# Compute R² and RMSE
		if n_features == 1:
			r2 = r2_score(real_padded.flatten(), sim_padded.flatten())
			rmse = np.sqrt(mean_squared_error(real_padded.flatten(), sim_padded.flatten()))
		else:
			r2 = np.mean([r2_score(real_padded[:, i], sim_padded[:, i]) for i in range(n_features)])
			rmse = np.mean([np.sqrt(mean_squared_error(real_padded[:, i], sim_padded[:, i])) for i in range(n_features)])

		# Update best metrics if better
		if (r2 > best_r2) or (r2 == best_r2 and rmse < best_rmse):
			best_r2 = r2
			best_rmse = rmse

	return best_r2, best_rmse
	
def compute_fitness(trace, petri_net):
	
	try:
		fitness = func_timeout.func_timeout(
			timeout=60,
			func=pm4py.fitness_alignments,
			args=[
				trace, 
				petri_net["network"], 
				petri_net["initial_marking"], 
				petri_net["final_marking"]
			]
		)
		# Extract the log fitness value
		fitness = fitness["log_fitness"]
	except:
		raise
	
	return fitness

def compute_f1(pairs):

	if not pairs:
		return float('nan')

	# Extract predictions and true labels
	y_pred = [p[0] for p in pairs]
	y_true = [p[1] for p in pairs]

	# Compute basic counts
	tp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 1)
	fp = sum(1 for p, t in zip(y_pred, y_true) if p == 1 and t == 0)
	fn = sum(1 for p, t in zip(y_pred, y_true) if p == 0 and t == 1)

	# Precision and recall
	precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
	recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

	# F1 score
	f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

	return f1

def compute_mean_evaluation_stats(evaluation_stats):

	if not evaluation_stats:
		return {'time': float('nan'),
				'fitness_ratio': float('nan'),
				'rmse_ratio': float('nan'),
				'r2_ratio': float('nan')}

	keys = ['time', 'fitness_ratio', 'rmse_ratio', 'r2_ratio']
	means = {}

	for key in keys:
		# Collect valid values (ignore NaN and inf)
		values = [
			entry[key] for entry in evaluation_stats
			if key in entry and not math.isnan(entry[key]) and not math.isinf(entry[key])
		]
		means[key] = sum(values) / len(values) if values else float('nan')

	return means

def write_metrics(f1_scores, evaluation_stats):


    for dataset_key in f1_scores.keys():  # e.g., 'V', 'W'
        dataset_folder = os.path.join(output_dir, dataset_key)
        os.makedirs(dataset_folder, exist_ok=True)

        for cluster_key in f1_scores[dataset_key].keys():  # e.g., '2', '3', '4'
            for pd_algorithm in f1_scores[dataset_key][cluster_key].keys():  # e.g., 'ilp75'

                # --- F1 mean and std ---
                f1_list = f1_scores[dataset_key][cluster_key][pd_algorithm]
                f1_mean = np.nan if not f1_list else np.mean(f1_list)
                f1_std = np.nan if not f1_list else np.std(f1_list)

                # --- Stats mean and std ---
                stats_list = evaluation_stats[dataset_key][cluster_key][pd_algorithm]
                keys = ['time', 'fitness_ratio', 'rmse_ratio', 'r2_ratio']
                mean_stats = {}
                std_stats = {}

                for key in keys:
                    values = [
                        d[key] for d in stats_list
                        if key in d and not math.isnan(d[key]) and not math.isinf(d[key])
                    ]
                    # Saturate ratios above 10.0 (not time)
                    if key != 'time':
                        values = [min(v, 10.0) for v in values]

                    mean_stats[key] = np.nan if not values else np.mean(values)
                    std_stats[key] = np.nan if not values else np.std(values)

                # --- Write to file ---
                filename = f"{cluster_key}_{pd_algorithm}.txt"
                filepath = os.path.join(dataset_folder, filename)
                with open(filepath, "w") as f:
                    f.write(f"f1: {f1_mean:.6f} ± {f1_std:.6f}\n")
                    f.write(f"time: {mean_stats['time']:.6f} ± {std_stats['time']:.6f}\n")
                    f.write(f"fitness_ratio: {mean_stats['fitness_ratio']:.6f} ± {std_stats['fitness_ratio']:.6f}\n")
                    f.write(f"rmse_ratio: {mean_stats['rmse_ratio']:.6f} ± {std_stats['rmse_ratio']:.6f}\n")
                    f.write(f"r2_ratio: {mean_stats['r2_ratio']:.6f} ± {std_stats['r2_ratio']:.6f}\n")

    return None

def print_dict_keys(d, indent=0):
    for key, value in d.items():
        print("  " * indent + str(key))
        if isinstance(value, dict):
            print_dict_keys(value, indent + 1)

training_set, test_set = read_training_test_set()

'''
print_dict_keys(training_set)
print_dict_keys(test_set)
'''

'''
for rep_number in test_set:
	for anomaly_type in test_set[rep_number]:
		for number_of_clusters in test_set[rep_number][anomaly_type]:
			for pd_algorithm in test_set[rep_number][anomaly_type][number_of_clusters]:
				print(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["EL"][1])
				print(test_set[rep_number][anomaly_type][number_of_clusters][pd_algorithm]["P"]["TS"][1])
'''				

f1_scores, evaluation_stats = classify_windows(training_set, test_set)
write_metrics(f1_scores, evaluation_stats)


