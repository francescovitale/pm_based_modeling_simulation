:: Options:
:: anomaly_type=[V, W]
:: pd_variant=[im, imf25, imf50, imf75, imf99, ilp, ilp25, ilp50, ilp75, ilp99, alpha]
:: n_clusters=<integer>
:: accuracy=<float>
:: n_simulation_traces=<integer>
:: clustering_technique=[kmeans, agglomerative, gmm]

set pd_variant=%1
set accuracy=%2
set anomaly_type=%3
set n_simulation_traces=300
set clustering_technique=kmeans
set synthetic_data=0
set n_sensor_types=1
set sensors=J2

call clean_environment_road_data
	
xcopy ROAD_Data\Output\WE\Training\%anomaly_type% Input\AD\Data /E
copy ROAD_Data\Output\WE\Test\%anomaly_type%\* Input\ELE\Data\Test
copy ROAD_Data\Output\WE\clustering_parameters.txt Input\PD
copy ROAD_Data\Output\WE\clustering_parameters.txt Input\S\Clustering

python anomaly_detection.py %accuracy%
	
copy Output\AD\Data\* Input\ELE\Data\Training
	
python event_log_extraction.py %synthetic_data% %n_sensor_types% %sensors%
	
copy Output\ELE\EventLog\EL_TR.xes Input\PD\EventLog
	
python process_discovery.py %pd_variant% %synthetic_data%
	
copy Output\PD\PetriNet\* Input\S\PetriNet
	
python simulation.py %n_simulation_traces% %synthetic_data%




