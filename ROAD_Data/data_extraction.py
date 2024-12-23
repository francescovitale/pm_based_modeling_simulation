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

input_dir = "Input/DE/"
output_dir = "Output/DE/"

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

def extract_normal_anomalous_data(training_observation, velocity_observation, sensor_types):

	observations = {}
	observations_columns = []
	if "S" in sensor_types:
		observations_columns = observations_columns + supply_columns
	for i in range(0,7):
		if "J"+str(i+1) in sensor_types:
			observations_columns = observations_columns + joint_columns[i]
			
	observations["T"] = get_observation(training_observation, sensor_types, training_datasets_columns, None, joint_columns, supply_columns, observations_columns)
	observations["V"] = get_observation(velocity_observation, sensor_types, None, other_datasets_columns, joint_columns, supply_columns, observations_columns)

	observations["T"] = observations["T"][240:]
	observations["T"] = observations["T"][:n_samples]
	observations["V"] = observations["V"][225:]
	observations["V"] = observations["V"][:n_samples]

	return observations["T"], observations["V"]

def save_behavior(normal_data, anomalous_data, training_percentage, test_percentage):

	n_normal_training_samples = int(len(normal_data)*training_percentage)
	n_anomalous_training_samples = int(len(anomalous_data)*training_percentage)

	n_normal_test_samples = len(normal_data) - n_normal_training_samples
	n_anomalous_test_samples = len(anomalous_data) - n_anomalous_training_samples

	training_normal_data = normal_data[0:n_normal_training_samples].copy()
	training_normal_data["Label"] = ["N"]*len(training_normal_data)
	training_anomalous_data = anomalous_data[0:n_anomalous_training_samples].copy()
	training_anomalous_data["Label"] = ["A"]*len(training_anomalous_data)
	training_data = pd.concat([training_normal_data, training_anomalous_data], ignore_index = True)

	test_normal_data = normal_data[n_normal_training_samples:(n_normal_training_samples + n_normal_test_samples)].copy()
	test_normal_data["Label"] = ["N"]*len(test_normal_data)
	test_anomalous_data = anomalous_data[n_anomalous_test_samples:(n_anomalous_training_samples + n_anomalous_test_samples)].copy()
	test_anomalous_data["Label"] = ["A"]*len(test_anomalous_data)
	test_data = pd.concat([test_normal_data, test_anomalous_data], ignore_index = True)

	training_data.to_csv(output_dir + "Training.csv", index = False)
	test_data.to_csv(output_dir + "Test.csv", index = False)

try:
	training_percentage = float(sys.argv[1])
	test_percentage = float(sys.argv[2])
	sensor_types = []
	n_sensors = int(sys.argv[3])
	for i in range(1,n_sensors+1):
		sensor_types.append(sys.argv[3+i])
except:
	print("Enter the correct number of input arguments.")
	sys.exit()

dataset = Dataset(normalize=False)
training_observations = dataset.sets['training']
velocity_observations = dataset.sets['velocity']

normal_data, anomalous_data = extract_normal_anomalous_data(training_observations[0], velocity_observations[0], sensor_types)
save_behavior(normal_data, anomalous_data, training_percentage, test_percentage)
