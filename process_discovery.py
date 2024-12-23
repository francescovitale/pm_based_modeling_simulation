from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

import pm4py
import sys
import os

import time

input_dir = "Input/PD/"
input_eventlog_dir = input_dir + "EventLog/"

output_dir = "Output/PD/"
output_timing_dir = output_dir + "Timing/"
output_petrinet_dir = output_dir + "PetriNet/"

variant = ""

def read_event_log():

	event_log = xes_importer.apply(input_eventlog_dir + "EL.xes")

	return event_log
	
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
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.00)

	elif variant == "ilp25":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.25)

	elif variant == "ilp50":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.50)

	elif variant == "ilp75":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.75)

	elif variant == "ilp99":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_ilp(event_log, alpha=0.99)

	elif variant == "alpha":
		petri_net["network"], petri_net["initial_marking"], petri_net["final_marking"] = pm4py.discover_petri_net_alpha(event_log)

	return petri_net

def export_petri_net(petri_net):
	pnml_exporter.apply(petri_net["network"], petri_net["initial_marking"], output_petrinet_dir + "PN.pnml", final_marking = petri_net["final_marking"])
	
def write_diagnoses_timing(mean_diagnoses_time, std_diagnoses_time, diagnoses_time_percentage, diagnoses_std_time_percentage):

	file = open(output_test_metrics_dir + "time.txt", "w")
	file.write("Mean time: " + str(mean_diagnoses_time) + "\n")
	file.write("Std time: " + str(std_diagnoses_time) + "\n")
	file.write("Mean time percentage: " + str(diagnoses_time_percentage) + "\n")
	file.write("Std time percentage: " + str(diagnoses_std_time_percentage))
	file.close()

	return None	
	
def write_process_discovery_timing(process_discovery_time):

	file = open(output_timing_dir + "pd_time.txt", "w")
	file.write(str(process_discovery_time))
	file.close()
	
try:
	variant = sys.argv[1]
except:
	print("Enter the right number of input arguments.")
	sys.exit()
	
	
event_log = read_event_log()
process_discovery_time = time.time()
petri_net = process_discovery(event_log, variant)
process_discovery_time = time.time() - process_discovery_time
export_petri_net(petri_net)
write_process_discovery_timing(process_discovery_time)