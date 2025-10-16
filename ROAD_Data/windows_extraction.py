import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import sys
import pandas as pd
import numpy as np
import math
import os
from RoADDataset import Dataset

from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky
from sklearn.model_selection import train_test_split

import random

input_dir = "Input/WE/"

output_dir = "Output/WE/"
output_training_dir = output_dir + "Training/"
output_test_dir = output_dir + "Test/"
output_normal_dir = output_dir + "Normal/"

n_clusters = -1
n_components = -1
clustering_technique = ""

training_datasets_columns = ["ID", "AP", "C", "F", "PA", "P", "PF", "RP", "V", "J1XA", "J1YA", "J1ZA", "J1XAV", "J1YAV", "J1ZAV", "J1QOC1", "J1QOC2", "J1QOC3", "J1QOC4", "J1T", "J2XA", "J2YA", "J2ZA", "J2XAV", "J2YAV", "J2ZAV", "J2QOC1", "J2QOC2", "J2QOC3", "J2QOC4", "J2T", "J3XA", "J3YA", "J3ZA", "J3XAV", "J3YAV", "J3ZAV", "J3QOC1", "J3QOC2", "J3QOC3", "J3QOC4", "J3T", "J4XA", "J4YA", "J4ZA", "J4XAV", "J4YAV", "J4ZAV", "J4QOC1", "J4QOC2", "J4QOC3", "J4QOC4", "J4T", "J5XA", "J5YA", "J5ZA", "J5XAV", "J5YAV", "J5ZAV", "J5QOC1", "J5QOC2", "J5QOC3", "J5QOC4", "J5T", "J6XA", "J6YA", "J6ZA", "J6XAV", "J6YAV", "J6ZAV", "J6QOC1", "J6QOC2", "J6QOC3", "J6QOC4", "J6T", "J7XA", "J7YA", "J7ZA", "J7XAV", "J7YAV", "J7ZAV", "J7QOC1", "J7QOC2", "J7QOC3", "J7QOC4", "J7T"]

other_datasets_columns = ["ID", "AP", "C", "F", "PA", "P", "PF", "RP", "V", "J1XA", "J1YA", "J1ZA", "J1XAV", "J1YAV", "J1ZAV", "J1QOC1", "J1QOC2", "J1QOC3", "J1QOC4", "J1T", "J2XA", "J2YA", "J2ZA", "J2XAV", "J2YAV", "J2ZAV", "J2QOC1", "J2QOC2", "J2QOC3", "J2QOC4", "J2T", "J3XA", "J3YA", "J3ZA", "J3XAV", "J3YAV", "J3ZAV", "J3QOC1", "J3QOC2", "J3QOC3", "J3QOC4", "J3T", "J4XA", "J4YA", "J4ZA", "J4XAV", "J4YAV", "J4ZAV", "J4QOC1", "J4QOC2", "J4QOC3", "J4QOC4", "J4T", "J5XA", "J5YA", "J5ZA", "J5XAV", "J5YAV", "J5ZAV", "J5QOC1", "J5QOC2", "J5QOC3", "J5QOC4", "J5T", "J6XA", "J6YA", "J6ZA", "J6XAV", "J6YAV", "J6ZAV", "J6QOC1", "J6QOC2", "J6QOC3", "J6QOC4", "J6T", "J7XA", "J7YA", "J7ZA", "J7XAV", "J7YAV", "J7ZAV", "J7QOC1", "J7QOC2", "J7QOC3", "J7QOC4", "J7T", "L"]

supply_columns = ["AP", "C", "F", "PA", "P", "PF", "RP", "V"]
	
'''joint_columns=[["J1XA", "J1YA", "J1ZA", "J1XAV", "J1YAV", "J1ZAV", "J1QOC1", "J1QOC2", "J1QOC3", "J1QOC4", "J1T"], 
	["J2XA", "J2YA", "J2ZA", "J2XAV", "J2YAV", "J2ZAV", "J2QOC1", "J2QOC2", "J2QOC3", "J2QOC4", "J2T"],
	["J3XA", "J3YA", "J3ZA", "J3XAV", "J3YAV", "J3ZAV", "J3QOC1", "J3QOC2", "J3QOC3", "J3QOC4", "J3T"],
	["J4XA", "J4YA", "J4ZA", "J4XAV", "J4YAV", "J4ZAV", "J4QOC1", "J4QOC2", "J4QOC3", "J4QOC4", "J4T"],
	["J5XA", "J5YA", "J5ZA", "J5XAV", "J5YAV", "J5ZAV", "J5QOC1", "J5QOC2", "J5QOC3", "J5QOC4", "J5T"],
	["J6XA", "J6YA", "J6ZA", "J6XAV", "J6YAV", "J6ZAV", "J6QOC1", "J6QOC2", "J6QOC3", "J6QOC4", "J6T"],
	["J7XA", "J7YA", "J7ZA", "J7XAV", "J7YAV", "J7ZAV", "J7QOC1", "J7QOC2", "J7QOC3", "J7QOC4", "J7T"]]'''

joint_columns=[["J1XA", "J1YA", "J1ZA"], 
	["J2XA", "J2YA", "J2ZA"],
	["J3XA", "J3YA", "J3ZA"],
	["J4XA", "J4YA", "J4ZA"],
	["J5XA", "J5YA", "J5ZA"],
	["J6XA", "J6YA", "J6ZA"],
	["J7XA", "J7YA", "J7ZA"]]

n_samples = 10000

def per_joint_state_extraction(training_observation, collision_observation, control_observation, weight_observation, velocity_observation, n_clusters, sensor_types):

	observations = {}
	observations_columns = []
	if "S" in sensor_types:
		observations_columns = observations_columns + supply_columns
	for i in range(0,7):
		if "J"+str(i+1) in sensor_types:
			observations_columns = observations_columns + joint_columns[i]
			
	observations["T"] = get_observation(training_observation, sensor_types, training_datasets_columns, None, joint_columns, supply_columns, observations_columns)
	observations["W"] = get_observation(weight_observation, sensor_types, None, other_datasets_columns, joint_columns, supply_columns, observations_columns)
	observations["V"] = get_observation(velocity_observation, sensor_types, None, other_datasets_columns, joint_columns, supply_columns, observations_columns)

	observations["T"] = observations["T"][240:]
	reuse_parameters = 0
	_, clustering_parameters = cluster_dataset(observations["T"], reuse_parameters, None)
	reuse_parameters = 1

	'''
	observations["T"].to_csv(output_dir + "training.csv", index = False)
	observations["W"].to_csv(output_dir + "weight.csv", index = False)
	observations["V"].to_csv(output_dir + "velocity.csv", index = False)
	'''

	observations["T"] = observations["T"][:n_samples]
	training_normal_v_windows, training_normal_w_windows = split_data(observations["T"], "T")
	observations["W"] = observations["W"][485:]
	observations["W"] = observations["W"][:n_samples]
	normal_w_windows, anomalous_w_windows = split_data(observations["W"], "W")
	observations["V"] = observations["V"][225:]
	observations["V"] = observations["V"][:n_samples]
	normal_v_windows, anomalous_v_windows = split_data(observations["V"], "V")
	
	# extracting the T windows
	for idx,df in enumerate(training_normal_v_windows):
		training_normal_v_windows[idx], _ = cluster_dataset(df, reuse_parameters, clustering_parameters)
		if "S" in sensor_types:
			training_normal_v_windows[idx].rename(columns={"Cluster": "CS"}, inplace=True)
		else:
			for i in range(0,7):
				if "J"+str(i+1) in sensor_types:
					training_normal_v_windows[idx].rename(columns={"Cluster": "C" + str(i+1)}, inplace=True)
	for idx,df in enumerate(training_normal_w_windows):
		training_normal_w_windows[idx], _ = cluster_dataset(df, reuse_parameters, clustering_parameters)
		if "S" in sensor_types:
			training_normal_w_windows[idx].rename(columns={"Cluster": "CS"}, inplace=True)
		else:
			for i in range(0,7):
				if "J"+str(i+1) in sensor_types:
					training_normal_w_windows[idx].rename(columns={"Cluster": "C" + str(i+1)}, inplace=True)	

	# extracting the W windows
	for idx,df in enumerate(normal_w_windows):
		normal_w_windows[idx], _ = cluster_dataset(df, reuse_parameters, clustering_parameters)
		if "S" in sensor_types:
			normal_w_windows[idx].rename(columns={"Cluster": "CS"}, inplace=True)
		else:
			for i in range(0,7):
				if "J"+str(i+1) in sensor_types:
					normal_w_windows[idx].rename(columns={"Cluster": "C" + str(i+1)}, inplace=True)
	for idx,df in enumerate(anomalous_w_windows):
		anomalous_w_windows[idx], _ = cluster_dataset(df, reuse_parameters, clustering_parameters)
		if "S" in sensor_types:
			anomalous_w_windows[idx].rename(columns={"Cluster": "CS"}, inplace=True)
		else:
			for i in range(0,7):
				if "J"+str(i+1) in sensor_types:
					anomalous_w_windows[idx].rename(columns={"Cluster": "C" + str(i+1)}, inplace=True)								
				
	# extracting the V windows			
	for idx,df in enumerate(normal_v_windows):
		normal_v_windows[idx], _ = cluster_dataset(df, reuse_parameters, clustering_parameters)
		if "S" in sensor_types:
			normal_v_windows[idx].rename(columns={"Cluster": "CS"}, inplace=True)
		else:
			for i in range(0,7):
				if "J"+str(i+1) in sensor_types:
					normal_v_windows[idx].rename(columns={"Cluster": "C" + str(i+1)}, inplace=True)
	for idx,df in enumerate(anomalous_v_windows):
		anomalous_v_windows[idx], _ = cluster_dataset(df, reuse_parameters, clustering_parameters)
		if "S" in sensor_types:
			anomalous_v_windows[idx].rename(columns={"Cluster": "CS"}, inplace=True)
		else:
			for i in range(0,7):
				if "J"+str(i+1) in sensor_types:
					anomalous_v_windows[idx].rename(columns={"Cluster": "C" + str(i+1)}, inplace=True)					
				

	return training_normal_v_windows, training_normal_w_windows, normal_w_windows, anomalous_w_windows, normal_v_windows, anomalous_v_windows, clustering_parameters

def get_observation(observation, sensor_types, training_datasets_columns, other_datasets_columns, joint_columns, supply_columns, columns):

	df = pd.DataFrame(columns = columns)
	if training_datasets_columns != None:
		temp = pd.DataFrame(columns = training_datasets_columns, data = observation)
	elif other_datasets_columns != None:
		temp = pd.DataFrame(columns = other_datasets_columns, data = observation)
	supply_df = temp[supply_columns]
	joint_dfs = []
	for idx,set in enumerate(joint_columns):
		joint_df = temp[set]
		joint_dfs.append(joint_df)
	if "S" in sensor_types:
		for column in supply_df:
			df[column] = supply_df[column]	
	for idx,joint_df in enumerate(joint_dfs):
		if "J"+str(idx+1) in sensor_types:
			for column in joint_df:
				df[column] = joint_df[column]
	
	return df

def cluster_dataset(dataset, reuse_parameters, clustering_parameters_in):
	clustered_dataset = dataset.copy()
	clustering_parameters = {}

	if reuse_parameters == 0:
		if clustering_technique == "gmm":
			gaussian_mixture = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0).fit(dataset)
			clustering_parameters["weights"] = gaussian_mixture.weights_
			clustering_parameters["means"] = gaussian_mixture.means_
			clustering_parameters["covariances"] = gaussian_mixture.covariances_
			labels_probability = gaussian_mixture.predict_proba(dataset)
			for j in range(n_clusters):
				temp = []
				for i in range(0,len(dataset)):
					temp.append(labels_probability[i][j])
				clustered_dataset["Cluster_" + str(j)] = temp
			cluster_ids = []
			for row_idx,row in clustered_dataset.iterrows():
				clusters = []
				for i in range(0, n_clusters):
					clusters.append(row["Cluster_" + str(i)])
				cluster_ids.append(clusters.index(max(clusters)))
			for i in range(0,n_clusters):
				clustered_dataset = clustered_dataset.drop(axis=1, columns="Cluster_"+str(i))
			clustered_dataset["Cluster"] = cluster_ids


		elif clustering_technique == "kmeans" or clustering_technique == "agglomerative":
			if clustering_technique == "agglomerative":
				cluster_configuration = AgglomerativeClustering(n_clusters=n_clusters, metric='cityblock', linkage='average')
				cluster_labels = cluster_configuration.fit_predict(clustered_dataset)
			elif clustering_technique == "kmeans":
				kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(clustered_dataset)
				cluster_labels = kmeans.labels_

			clustered_dataset["Cluster"] = cluster_labels
			cluster_labels = cluster_labels.tolist()
			used = set();
			clusters = [x for x in cluster_labels if x not in used and (used.add(x) or True)]

			instances_sets = {}
			centroids = {}
			
			for cluster in clusters:
				instances_sets[cluster] = []
				centroids[cluster] = []
				
			
			temp = clustered_dataset
			for index, row in temp.iterrows():
				instances_sets[int(row["Cluster"])].append(row.values.tolist())
			
			n_features_per_instance = len(instances_sets[0][0])-1
			
			for instances_set_label in instances_sets:
				instances = instances_sets[instances_set_label]
				for idx, instance in enumerate(instances):
					instances[idx] = instance[0:n_features_per_instance]
				for i in range(0,n_features_per_instance):
					values = []
					for instance in instances:
						values.append(instance[i])
					centroids[instances_set_label].append(np.mean(values))
					
			clustering_parameters = centroids
			
	else:

		if clustering_technique == "gmm":
			gaussian_mixture = GaussianMixture(n_components=n_clusters, covariance_type='full')
			gaussian_mixture.weights_ = clustering_parameters_in["weights"]
			gaussian_mixture.means_ = clustering_parameters_in["means"]
			gaussian_mixture.covariances_ = clustering_parameters_in["covariances"]
			gaussian_mixture.precisions_cholesky_ = _compute_precision_cholesky(clustering_parameters_in["covariances"], 'full')
			labels_probability = gaussian_mixture.predict_proba(dataset)
			for j in range(n_clusters):
				temp = []
				for i in range(0,len(dataset)):
					temp.append(labels_probability[i][j])
				clustered_dataset["Cluster_" + str(j)] = temp
			cluster_ids = []
			for row_idx,row in clustered_dataset.iterrows():
				clusters = []
				for i in range(0, n_clusters):
					clusters.append(row["Cluster_" + str(i)])
				cluster_ids.append(clusters.index(max(clusters)))
			for i in range(0,n_clusters):
				clustered_dataset = clustered_dataset.drop(axis=1, columns="Cluster_"+str(i))
			clustered_dataset["Cluster"] = cluster_ids

		elif clustering_technique == "kmeans" or clustering_technique == "agglomerative":
			clusters = []
			for index, instance in clustered_dataset.iterrows():
				min_value = float('inf')
				min_centroid = -1
				for centroid in clustering_parameters_in:
					centroid_coordinates = np.array([float(i) for i in clustering_parameters_in[centroid]])
					dist = np.linalg.norm(instance.values-centroid_coordinates)
					if dist<min_value:
						min_value = dist
						min_centroid = centroid
				clusters.append(min_centroid)
			
			clustered_dataset["Cluster"] = clusters

	return clustered_dataset, clustering_parameters

def split_data(timeseries, observation_type):
	
	
	if observation_type == "V":
		normal_windows = []
		anomalous_windows = []
		normal_v_pairs, anomalous_v_pairs = build_pairs(observation_type)
		for pair in normal_v_pairs:
			normal_windows.append(timeseries[pair[0]:pair[1]])
		for pair in anomalous_v_pairs:
			anomalous_windows.append(timeseries[pair[0]:pair[1]])
		return normal_windows, anomalous_windows	
			
	elif observation_type == "W":
		normal_windows = []
		anomalous_windows = []
		normal_w_pairs, anomalous_w_pairs = build_pairs(observation_type)
		for pair in normal_w_pairs:
			normal_windows.append(timeseries[pair[0]:pair[1]])
		for pair in anomalous_w_pairs:
			anomalous_windows.append(timeseries[pair[0]:pair[1]])	
		return normal_windows, anomalous_windows	
			
	elif observation_type == "T":
		normal_v_windows = []
		normal_w_windows = []
		normal_training_v_pairs, normal_training_w_pairs = build_pairs(observation_type)
		for pair in normal_training_v_pairs:
			normal_v_windows.append(timeseries[pair[0]:pair[1]])
		for pair in normal_training_w_pairs:
			normal_w_windows.append(timeseries[pair[0]:pair[1]])
		return normal_v_windows, normal_w_windows

def build_pairs(sample_type):

	normal_pairs = []
	anomalous_pairs = []
	
	if sample_type == "V":
		anomalous_pairs = [
		[30,135], [180,295], [345,425], [480,580], [630,730], [780,880], [925,990], [1035,1095], [1150,1210], [1260,1335], [1385,1450], [1495,1595], [1650,1725], [1770,1885], [1940,2015], [2060,2115], [2160,2220], [2260,2315], [2360,2415], [2460,2515], [2555,2630], [2670,2760], [2795,2870], [2925,2990], [3040,3110], [3165,3230], [3280,3335], [3390,3450], [3500,3555], [3610,3670], [3720,3780], [3830,3905], [3955,4040], [4090,4170], [4225,4300], [4350,4425], [4485,4605], [4660,4720], [4770,4830], [4880,4940], [4990,5050], [5100,5160], [5210,5280], [5330,5410], [5450,5520], [5570,5645], [5690,5785], [5830,5930], [5980,6040], [6090,6190], [6245,6300], [6340,6400], [6440,6500], [6545,6650], [6700,6770], [6825,6940], [6985,7055], [7100,7200], [7250,7325], [7375,7430], [7485,7580], [7630,7735], [7785,7840], [7895,7950], [8000,8080], [8135,8210], [8265,8340], [8400,8470], [8525,8600], [8645,8720], [8765,8825], [8880,8930], [8990,9070], [9115,9170], [9225,9280], [9335,9445], [9495,9565], [9615,9685], [9740,9805], [9860,9930] 
		]
		normal_pairs = get_complementary_pairs(anomalous_pairs)
		
		return normal_pairs, anomalous_pairs
	elif sample_type == "W":
		anomalous_pairs = [
		[95,215], [380,515], [690,790], [845,940], [990,1095], [1145,1245], [1300,1420], [1475,1570], [1745,1830], [2010,2100], [2270,2375], [2445,2595], [2665,2725], [2775,2945], [3015,3085], [3135, 3275], [3475,3635], [3835,3995], [4195,4355], [4450,4585], [4630,4760], [4850,4980], [5025,5150], [5240,5385], [5575,5675], [5670,6000], [6190,6300], [6455,6575], [6735,6830], [6890,6985], [7740,7910], [7980,8085], [8140,8310], [8380,8465], [8515,8680], [8870,9045], [9245,9420], [9625,9795], [9890,10000]
		]
		normal_pairs = get_complementary_pairs(anomalous_pairs)
		return normal_pairs, anomalous_pairs
	elif sample_type == "T":
		normal_v_pairs = [
		[25,90], [135,195], [245,310], [355,415], [465,530], [570,620], [665,710], [755,800], [845,890], [935,980], [1025,1085], [1140,1200], [1245,1300], [1360,1415], [1465,1525], [1575,1635], [1685,1735], [1775,1825], [1865,1915], [1950,2005], [2040,2100], [2130,2200], [2250,2310], [2350,2420], [2470,2530], [2570,2635], [2690,2755], [2795,2845], [2885,2935], [2975,3020], [3065,3115], [3155,3205], [3250,3310], [3350,3410], [3445,3505], [3540,3610], [3650,3720], [3770,3830], [3865,3920], [3955,4005], [4045,4095], [4130,4185], [4220,4275], [4315,4375], [4415,4480], [4515,4580], [4620,4675], [4715,4770], [4815,4880], [4920,4965], [5005,5060], [5095,5145], [5185,5240], [5275,5325], [5365,5435], [5475,5540], [5585,5655], [5695,5765], [5810,5870], [5915,5985], [5915,5980], [6025,6075], [6115,6170], [6200,6260], [6295,6350], [6385,6440], [6475,6540], [6575,6640], [6675,6740], [6775,6840], [6870,6940], [6975,7045], [7075,7130], [7160,7215], [7250,7310], [7345,7400], [7435,7485], [7525,7585], [7630,7685], [7725,7785], [7830,7890], [7925,7990], [7925,7985], [8030,8090], [8130,8180], [8215,8270], [8305,8360], [8400,8450], [8485,8535], [8575,8640], [8690,8755], [8795,8860], [8910,8970], [9020,9080], [9130,9190], [9240,9290], [9330,9380], [9430,9475], [9515,9570], [9610,9655], [9695,9760], [9800,9855], [9900,9955]
		]
		normal_w_pairs = get_complementary_pairs(normal_v_pairs)
		return normal_v_pairs, normal_w_pairs 	
	
def get_complementary_pairs(pairs):

	complementary_pairs = []
	
	total_range = [pairs[0][0], pairs[-1][1]]
	
	prev_end = total_range[0]

	for start, end in pairs:
		if start > prev_end:
			complementary_pairs.append([prev_end, start])
		prev_end = max(prev_end, end)

	if prev_end < total_range[1]:
		complementary_pairs.append([prev_end, total_range[1]])

	return complementary_pairs

def build_training_test_set(anomalous_w_windows, anomalous_v_windows, test_percentage):

	training_set = {}
	test_set = {}
	
	# W data
	training_set["W"], test_set["W"] = train_test_split(anomalous_w_windows, test_size=test_percentage)
	
	# V data
	training_set["V"], test_set["V"] = train_test_split(anomalous_v_windows, test_size=test_percentage)

	return training_set, test_set

def save_clustering_parameters(clustering_parameters):
	
	file = open(output_dir + "clustering_parameters.txt", "w")

	if clustering_technique == "gmm":
		for idx, centroid in enumerate(clustering_parameters["means"]):
			if idx<len(clustering_parameters["means"]):
				file.write(str(idx) + ":" + str(list(centroid)) + "\n")
			else:
				file.write(str(idx) + ":" + str(list(centroid)))

	elif clustering_technique == "kmeans" or clustering_technique == "agglomerative":
		for idx, clustering_parameter in enumerate(clustering_parameters):
			if idx<len(clustering_parameters)-1:
				file.write(str(clustering_parameter) + ":" + str(clustering_parameters[clustering_parameter]) + "\n")
			else:
				file.write(str(clustering_parameter) + ":" + str(clustering_parameters[clustering_parameter]))

	return None

def save_windows(training_set, test_set, normal_w_windows, normal_v_windows, training_normal_v_windows, training_normal_w_windows):
		
	# training set	
	
	isExist = os.path.exists(output_training_dir + "V")
	if not isExist:
		os.makedirs(output_training_dir + "V/N")
		os.makedirs(output_training_dir + "V/A")
		
	for idx,window in enumerate(normal_v_windows):
		window.to_csv(output_training_dir + "V/N/WNDW_" + str(idx) + ".csv", index = False)
	for idx,window in enumerate(training_set["V"]):
		window.to_csv(output_training_dir + "V/A/WNDW_" + str(idx) + ".csv", index = False)	
		
	isExist = os.path.exists(output_training_dir + "W")
	if not isExist:
		os.makedirs(output_training_dir + "W/N")
		os.makedirs(output_training_dir + "W/A")
	
	for idx,window in enumerate(normal_w_windows):
		window.to_csv(output_training_dir + "W/N/WNDW_" + str(idx) + ".csv", index = False)	
	for idx,window in enumerate(training_set["W"]):
		window.to_csv(output_training_dir + "W/A/WNDW_" + str(idx) + ".csv", index = False)	
	
	# test set
	
	isExist = os.path.exists(output_test_dir + "V")
	if not isExist:
		os.makedirs(output_test_dir + "V")
		
	isExist = os.path.exists(output_test_dir + "W")
	if not isExist:
		os.makedirs(output_test_dir + "W")
	
	for idx,window in enumerate(test_set["V"]):
		window.to_csv(output_test_dir + "V/WNDW_" + str(idx) + ".csv", index = False)
	for idx,window in enumerate(test_set["W"]):
		window.to_csv(output_test_dir + "W/WNDW_" + str(idx) + ".csv", index = False)
		
	# normal set
	
	isExist = os.path.exists(output_normal_dir + "V")
	if not isExist:
		os.makedirs(output_normal_dir + "V")
		
	isExist = os.path.exists(output_normal_dir + "W")
	if not isExist:
		os.makedirs(output_normal_dir + "W")	

	for idx,window in enumerate(training_normal_v_windows):
		window.to_csv(output_normal_dir + "V/WNDW_" + str(idx) + ".csv", index = False)
	for idx,window in enumerate(training_normal_w_windows):
		window.to_csv(output_normal_dir + "W/WNDW_" + str(idx) + ".csv", index = False)
	
try:
	sensor_types = []
	clustering_technique = sys.argv[1]
	n_clusters = int(sys.argv[2])
	test_percentage = float(sys.argv[3])
	n_sensors = int(sys.argv[4])
	for i in range(1,n_sensors+1):
		sensor_types.append(sys.argv[4+i])
except:
	print("Enter the correct number of input arguments.")
	sys.exit()

dataset = Dataset(normalize=True)
training_observations = dataset.sets['training']
collision_observations = dataset.sets['collision']
control_observations = dataset.sets['control']
weight_observations = dataset.sets['weight']
velocity_observations = dataset.sets['velocity']

training_normal_v_windows, training_normal_w_windows, normal_w_windows, anomalous_w_windows, normal_v_windows, anomalous_v_windows, clustering_parameters = per_joint_state_extraction(training_observations[0], collision_observations[0], control_observations[0], weight_observations[0], velocity_observations[0], n_clusters, sensor_types)
training_normal_v_windows = random.sample(training_normal_v_windows, int((1-test_percentage)*len(training_normal_v_windows)))
training_normal_w_windows = random.sample(training_normal_w_windows, int((1-test_percentage)*len(training_normal_w_windows)))
training_set, test_set = build_training_test_set(anomalous_w_windows, anomalous_v_windows, test_percentage)
save_clustering_parameters(clustering_parameters)
save_windows(training_set, test_set, normal_w_windows, normal_v_windows, training_normal_v_windows, training_normal_w_windows)
