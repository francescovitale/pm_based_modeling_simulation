# Requirements to run the methodology

This project has been executed on a Windows 10 machine with Python 3.11.5. A few libraries have been used within Python modules. Among these, there are:

- pm4py 2.7.11.11
- scipy 1.14.0
- scikit-learn 1.3.0
- tslearn: 0.6.3

Please note that the list above is not comprehensive and there could be other requirements for running the project, including the RoAD dataset (https://gitlab.com/AlessioMascolini/roaddataset/).

# Execution instructions and project description

The methodology runs by executing the DOS experimentation_road_data.bat script on a Windows 10 machine with the requirements above. This script includes a set of parameters that can be customized to set: 

- The process discovery variant (pd_variant)
- The number of clusters (n_clusters)
- The anomaly type (anomaly_type)
- The number of simulation traces (n_simulation_traces)
- The number of K-shape clusters to isolate anomalous time series windows (n_ts_clusters)
- The clustering technique for event log extraction (clustering_technique)
- The simulation technique (simulation_technique)
- The number of experiment repetitions (n_reps)

For each experiment repetition, each pair of factor values is chosen, namely the number of clusters and the process discovery technique. Given a pair of factor values, the methodology:

1) Extracts the time series windows from RoAD data (see the windows_extraction.bat script under the ROAD_Data folder for further insight);
2) Isolates anomalous time series windows (correlate_timeseries.py);
3) Extracts anomalous event logs associated with the anomalous time series windows (event_log_extraction.py);
4) Builds the anomalous Petri net (process_discovery.py);
5) Simulates the anomalous Petri net (simulation.py).

When the experiment repetition is completed, the evaluation.py script executes to:

- Save the event logs for each number of clusters value;
- Save the anomalous Petri net for each number of clusters and process discovery technique pair;
- Calculate the quality metrics related to the modeling and simulation quality, i.e., the fitness, precision and conformance checking time metrics;
- Find the best simulated time series and compute the root mean squared error across all the simulated time series.

All the results can be found under the Results folder.
