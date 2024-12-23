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

import random

input_dir = "Input/WE/"
output_dir = "Output/WE/"

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
n_samples_per_window = 100
n_windows_to_save = 100

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

def per_joint_state_extraction(training_observation, collision_observation, control_observation, weight_observation, velocity_observation, n_samples_per_window, n_clusters, sensor_types):

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

	observations["T"] = observations["T"][:n_samples]
	observations["T"] = split_data(observations["T"], n_samples_per_window, "T")
	observations["W"] = observations["W"][485:]
	observations["W"] = observations["W"][:n_samples]
	observations["W"] = split_data(observations["W"], n_samples_per_window, "W")
	observations["V"] = observations["V"][225:]
	observations["V"] = observations["V"][:n_samples]
	observations["V"] = split_data(observations["V"], n_samples_per_window, "V")

	

	for idx,df in enumerate(observations["T"]):
		observations["T"][idx], _ = cluster_dataset(df, reuse_parameters, clustering_parameters)
		if "S" in sensor_types:
			observations["T"][idx].rename(columns={"Cluster": "CS"}, inplace=True)
		else:
			for i in range(0,7):
				if "J"+str(i+1) in sensor_types:
					observations["T"][idx].rename(columns={"Cluster": "C" + str(i+1)}, inplace=True)
				
	for idx,df in enumerate(observations["W"]):
		observations["W"][idx], _ = cluster_dataset(df, reuse_parameters, clustering_parameters)
		if "S" in sensor_types:
			observations["W"][idx].rename(columns={"Cluster": "CS"}, inplace=True)
		else:
			for i in range(0,7):
				if "J"+str(i+1) in sensor_types:
					observations["W"][idx].rename(columns={"Cluster": "C" + str(i+1)}, inplace=True)
					
	for idx,df in enumerate(observations["V"]):
		observations["V"][idx], _ = cluster_dataset(df, reuse_parameters, clustering_parameters)	
		if "S" in sensor_types:
			observations["V"][idx].rename(columns={"Cluster": "CS"}, inplace=True)
		else:
			for i in range(0,7):
				if "J"+str(i+1) in sensor_types:
					observations["V"][idx].rename(columns={"Cluster": "C" + str(i+1)}, inplace=True)

	return observations, clustering_parameters

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

def split_data(timeseries, n_samples_per_window, observation_type):

	windows = []
	temp_timeseries = timeseries.copy()

	n_windows = math.ceil(len(timeseries)/n_samples_per_window)
	for i in range(0,n_windows):
		windows.append(timeseries.head(n_samples_per_window))
		timeseries = timeseries.iloc[n_samples_per_window:]
	
	#windows = random.choices(windows, n_windows_to_save)
		

	'''
	if observation_type == "T":
		start_end_idx = [[0, 110], [110, 220], [220, 330], [330, 440], [440, 550], [550, 640], [640, 730], [730, 830], [830, 910], [910, 1000], [1000, 1110], [1110, 1220], [1220, 1330], [1330, 1440], [1440, 1550], [1550, 1660], [1660, 1750], [1750, 1845], [1845, 1930], [1930, 2025]]
		for pair in start_end_idx:
			windows.append(timeseries[pair[0]:pair[1]])
	elif observation_type == "V":
		start_end_idx = [[0, 150],[150, 325],[325, 455],[455, 605],[605, 750],[750, 905],[950, 1010],[1010, 1115],[1115, 1230],[1230, 1360],[1360, 1475], [1475, 1620], [1620, 1745], [1745, 1915], [1915, 2035], [2035, 2140], [2140, 2235], [2235, 2345], [2345, 2435], [2435,2540]]
		for pair in start_end_idx:
			windows.append(timeseries[pair[0]:pair[1]])
	elif observation_type == "W":
		start_end_idx = [[0, 175], [175, 305], [305, 480], [480, 600], [600, 760], [760, 910], [910, 1060], [1060, 1215], [1215, 1380], [1380, 1550], [1550, 1660], [1660, 1800], [1800, 1920], [1920, 2060], [2060, 2175], [2175, 2350], [2350, 2570], [2570, 2700], [2700, 2920], [2920, 3015]]
		for pair in start_end_idx:
			windows.append(timeseries[pair[0]:pair[1]])		
	'''
	
	return windows

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

def save_behavior(observations):

	for observation_type in observations:
		isExist = os.path.exists(output_dir + observation_type)
		if not isExist:
			os.makedirs(output_dir + observation_type)
		for idx,window in enumerate(observations[observation_type]):
			window.to_csv(output_dir + observation_type + "/" + "WNDW_" + str(idx) + ".csv", index = False)	
	
try:
	sensor_types = []
	clustering_technique = sys.argv[1]
	n_clusters = int(sys.argv[2])
	n_sensors = int(sys.argv[3])
	for i in range(1,n_sensors+1):
		sensor_types.append(sys.argv[3+i])
except:
	print("Enter the correct number of input arguments.")
	sys.exit()

dataset = Dataset(normalize=True)
training_observations = dataset.sets['training']
collision_observations = dataset.sets['collision']
control_observations = dataset.sets['control']
weight_observations = dataset.sets['weight']
velocity_observations = dataset.sets['velocity']

observations, clustering_parameters = per_joint_state_extraction(training_observations[0], collision_observations[0], control_observations[0], weight_observations[0], velocity_observations[0], n_samples_per_window, n_clusters, sensor_types)

save_clustering_parameters(clustering_parameters)
save_behavior(observations)
