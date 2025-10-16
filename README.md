# Requirements to run the methodology

This project has been executed on a Windows 10 machine with Python 3.11.5. A few libraries have been used within Python modules. Among these, there are:

- pm4py 2.7.11.11
- scipy 1.14.0
- scikit-learn 1.3.0
- tslearn: 0.6.3

Please note that the list above is not comprehensive and there could be other requirements for running the project, including the RoAD dataset (https://gitlab.com/AlessioMascolini/roaddataset/).

# Execution instructions and project description

The methodology runs by executing the DOS experimentation_road_data.bat script. This script includes experimental parameters to set: 

- The process discovery variant (pd_variant)
- The number of clusters (n_clusters)
- The anomaly type (anomaly_type)
- The number of simulation traces (n_simulation_traces)
- The clustering technique for event log extraction (clustering_technique)
- The simulation technique (simulation_technique)
- The number of experiment repetitions (n_reps)

For each experiment repetition, each pair of factor values is chosen, namely the number of clusters and the process discovery technique. Given a pair of factor values, the methodology:

1) Extracts the time series windows from RoAD data (see the windows_extraction.bat script under the ROAD_Data folder for further insight);
2) Isolates anomalous time series windows (anomaly_detection.py);
3) Extracts anomalous event logs associated with the anomalous time series windows (event_log_extraction.py);
4) Builds the anomalous Petri net (process_discovery.py);
5) Simulates the anomalous Petri net (simulation.py).

All the results can be found under the Results folder.

# Individual experiments

The data obtained from the execution of the framework can be used to perform different experiments. We have performed two experiments: the modeling and simulation experiment and the fault identification experiment

## Modeling and simulation experiment

This experiment aims to evaluate the RMSE, R^2 and arc-degree simplicity from the Petri nets obtained with a specific process discovery algorithm-number of clusters combination. To run the experiment, the content of the ModelingSimulationExperiment folder can be cloned and the evaluation.py script executed. The results can be found in the ModelingSimulationExperiment/Output folder.

## Fault identification experiment

This experiment aims to evaluate the fault identification capabilities of the method in terms of F1-score and time performance. As before, this evaluation is done for each specific process discovery algorithm-number of clusters combination. To run  the experiment, the content of the FaultIdentificationExperiment folder can be cloned and the evaluation.py script executed. The results can be found in the ModelingSimulationExperiment/Output folder.

Additionally, we have performed an ablation test to verify the impact of anomaly detection accuracy on the subsequent fault identification part of the method. We have evaluated this ability through the evaluation.py script in the AblationStudyPM. The results can be found in the AblationStudyPM/Output folder. We have also performed a comparison with other deep learning-based methods. The Jupyter Notebook fault_classification.ipynb within the AblationStudyDL can be executed on any Jupyter-compliant environment. The results can be found in the AblationStudyDL/Output folder.
