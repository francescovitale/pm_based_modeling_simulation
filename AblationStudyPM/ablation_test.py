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

def extract_acc(name):
    try:
        return float(name.split("_")[1])
    except:
        return 0.0  # fallback

def read_data():

	data = {}
	
	for rep_number in os.listdir(input_dir):
		data[rep_number] = {}

		rep_path = os.path.join(input_dir, rep_number)
		keys = sorted(os.listdir(rep_path), key=extract_acc)

		for key in keys:
			anomaly_type = key.split("_")[0]
			accuracy = key.split("_")[1]
		
			data[rep_number][key] = {}
			data[rep_number][key]["Training"] = {}
			
			# positive
			data[rep_number][key]["Training"]["P"] = {}
			temp_pn = {}
			temp_pn["network"], temp_pn["initial_marking"], temp_pn["final_marking"] = pnml_importer.apply(input_dir + rep_number + "/" + key + "/PetriNet/PN.pnml")
			data[rep_number][key]["Training"]["P"]["PN"] = temp_pn
			
			data[rep_number][key]["Training"]["P"]["TS"] = []
			simulated_ts_dir = os.path.join(input_dir, rep_number, key, "SimulatedTimeSeries")
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
			data[rep_number][key]["Training"]["P"]["TS"] = temp_simulated_ts
			
			# negative
			data[rep_number][key]["Training"]["N"] = {}
			data[rep_number][key]["Training"]["N"]["PN"] = {}
			if anomaly_type == "V":
				neg_config = f"W_{accuracy}"
			else:
				neg_config = f"V_{accuracy}"
			pnml_dir = os.path.join(input_dir, rep_number, neg_config, "PetriNet", "PN.pnml")
			temp_pn = {}
			temp_pn["network"], temp_pn["initial_marking"], temp_pn["final_marking"] = pnml_importer.apply(pnml_dir)
			data[rep_number][key]["Training"]["N"]["PN"] = temp_pn
			
			data[rep_number][key]["Training"]["N"]["TS"] = []
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
			data[rep_number][key]["Training"]["N"]["TS"] = temp_simulated_ts
			
			# positive
			data[rep_number][key]["Test"] = {}
			data[rep_number][key]["Test"]["P"] = {}
			data[rep_number][key]["Test"]["P"]["TS"] = []
			ts_dir = os.path.join(input_dir, rep_number, key, "TestTimeSeries")
			files = sorted(
				[f for f in os.listdir(ts_dir) if f.startswith("WNDW_") and f.endswith(".csv")],
				key=lambda x: int(x.split('_')[1].split('.')[0])
			)
			temp_ts = [pd.read_csv(os.path.join(ts_dir, fname)).drop("C2", axis=1) for fname in files]
			data[rep_number][key]["Test"]["P"]["TS"] = temp_ts
			pos_el_path = os.path.join(input_dir, rep_number, key, "TestEventLog", "EL_TST.xes")
			data[rep_number][key]["Test"]["P"]["EL"] = xes_importer.apply(pos_el_path)

			# negative
			data[rep_number][key]["Test"]["N"] = {}
			data[rep_number][key]["Test"]["N"]["TS"] = []
			ts_dir = os.path.join(input_dir, rep_number, neg_config, "TestTimeSeries")
			files = sorted(
				[f for f in os.listdir(ts_dir) if f.startswith("WNDW_") and f.endswith(".csv")],
				key=lambda x: int(x.split('_')[1].split('.')[0])
			)
			temp_ts = [pd.read_csv(os.path.join(ts_dir, fname)).drop("C2", axis=1) for fname in files]
			data[rep_number][key]["Test"]["N"]["TS"] = temp_ts
			neg_el_path = os.path.join(input_dir, rep_number, neg_config, "TestEventLog", "EL_TST.xes")
			data[rep_number][key]["Test"]["N"]["EL"] = xes_importer.apply(neg_el_path)
			

	return data

def classify_windows(data):

	f1_score = {}
	
	for key in data[list(data.keys())[0]]:
		f1_score[key] = []
	
		
	for rep_number in data:
		for key in data[rep_number]:
			classifications = []
			if len(data[rep_number][key]["Test"]["P"]["TS"]) != len(data[rep_number][key]["Test"]["P"]["EL"]):
				data[rep_number][key]["Test"]["P"]["EL"] = fill_missing_cases(data[rep_number][key]["Test"]["P"]["EL"])
			if len(data[rep_number][key]["Test"]["N"]["TS"]) != len(data[rep_number][key]["Test"]["N"]["EL"]):
				data[rep_number][key]["Test"]["N"]["EL"] = fill_missing_cases(data[rep_number][key]["Test"]["N"]["EL"])
			
			try:
				for trace, ts in zip(data[rep_number][key]["Test"]["P"]["EL"], data[rep_number][key]["Test"]["P"]["TS"]):
					elapsed = 0.0
								
					fitness_p = compute_fitness([trace], data[rep_number][key]["Training"]["P"]["PN"])
					fitness_n = compute_fitness([trace], data[rep_number][key]["Training"]["N"]["PN"])
								
					r2_p, rmse_p = get_best_simulated_metrics(ts,data[rep_number][key]["Training"]["P"]["TS"])
					r2_n, rmse_n = get_best_simulated_metrics(ts,data[rep_number][key]["Training"]["N"]["TS"])
								
					votes = []
					votes.append("P" if rmse_p < rmse_n else "N")
					votes.append("P" if r2_p > r2_n else "N")
					votes.append("P" if fitness_p > fitness_n else "N")
					classified_type = 1 if votes.count("P") > votes.count("N") else 0
					classifications.append([classified_type, 1])
							
				for trace, ts in zip(data[rep_number][key]["Test"]["N"]["EL"], data[rep_number][key]["Test"]["N"]["TS"]):
						
					fitness_p = compute_fitness([trace], data[rep_number][key]["Training"]["P"]["PN"])
					fitness_n = compute_fitness([trace], data[rep_number][key]["Training"]["N"]["PN"])
							
					r2_p, rmse_p = get_best_simulated_metrics(ts,data[rep_number][key]["Training"]["P"]["TS"])
					r2_n, rmse_n = get_best_simulated_metrics(ts,data[rep_number][key]["Training"]["N"]["TS"])
					votes = []
					votes.append("P" if rmse_p < rmse_n else "N")
					votes.append("P" if r2_p > r2_n else "N")
					votes.append("P" if fitness_p > fitness_n else "N")
					classified_type = 1 if votes.count("P") > votes.count("N") else 0
					classifications.append([classified_type, 0])
								
			except:
				classifications = []
						
			f1_score[key].append(compute_f1(classifications))

	return f1_score

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

def save_mean_std_results(results_dict):

    lines = []
    for key, values in results_dict.items():
        arr = np.array(values, dtype=float)
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        lines.append(f"{key}: {mean_val:.4f} ± {std_val:.4f}")
    
    # Write to txt file
    with open(output_dir + "f1_scores_pm.txt", "w") as f:
        f.write("\n".join(lines))
    
    print(f"✅ Results saved to {output_dir}")

data = read_data()
f1_score = classify_windows(data)
save_mean_std_results(f1_score)