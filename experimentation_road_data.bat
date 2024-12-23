:: Options:
:: anomaly_type=[V, W]; one anomaly type at a time
:: pd_variant=[im, imf25, imf50, imf75, imf99, ilp, ilp25, ilp50, ilp75, ilp99, alpha]
:: n_clusters=<integer>
:: n_simulation_traces=<integer>
:: clustering_technique=[kmeans, agglomerative, gmm]
:: n_ts_clusters=<integer>
:: simulation_technique=[process_mining]

set pd_variant=ilp25 imf25
set n_clusters=3 5 7 10 12 15
set anomaly_type=W
set n_simulation_traces=300
set clustering_technique=kmeans
set n_ts_clusters=5
set simulation_technique=process_mining
set n_reps=10

for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /l %%a in (1, 1, %n_reps%) do (
	for %%x in (%clustering_technique%) do (

		cd Evaluation
		call clean_environment
		cd ..
		
		mkdir Results\%%x_%%a
		mkdir Results\%%x_%%a\EventLogs
		mkdir Results\%%x_%%a\PetriNets
		mkdir Results\%%x_%%a\Metrics
		mkdir Results\%%x_%%a\TimeSeriesPairs

		for %%y in (%n_clusters%) do (
				
			cd ROAD_Data
			call windows_extraction %%x %%y
			cd ..

			call clean_environment
			mkdir Evaluation\Input\TimeSeries\%%y
			
			copy ROAD_Data\Output\WE\%anomaly_type%\* Input\CT\Data
			copy ROAD_Data\Output\WE\clustering_parameters.txt Input\S\Clustering

			python correlate_timeseries.py %n_ts_clusters%
			
			copy Output\CT\Data\* Evaluation\Input\TimeSeries\%%y
			copy Output\CT\Data\* Input\ELE\Data

			python event_log_extraction.py 0 1 J2

			copy Output\ELE\EventLog\* Evaluation\Input\EventLogs
			ren Evaluation\Input\EventLogs\EL.xes %%y.xes
					
			copy Output\ELE\EventLog\* Input\PD\EventLog
			copy Output\ELE\EventLog\* Input\S\EventLog

			for %%z in (%pd_variant%) do (

				python process_discovery.py %%z

				copy Output\PD\Timing\* Results\%%x_%%a\Metrics
				ren Results\%%x_%%a\Metrics\pd_time.txt %%y_%%z_pd_time.txt 
				copy Output\PD\PetriNet\* Evaluation\Input\PetriNets
				ren Evaluation\Input\PetriNets\PN.pnml %%y_%%z.pnml
				copy Output\PD\PetriNet Input\S\PetriNet
					
				python simulation.py %n_simulation_traces% 0 %simulation_technique%
				
				copy Output\S\Timing\* Results\%%x_%%a\Metrics
				ren Results\%%x_%%a\Metrics\simulation_time.txt %%y_%%z_simulation_time.txt 
				mkdir Evaluation\Input\SimulatedTimeSeries\%%y_%%z
				copy Output\S\Data\* Evaluation\Input\SimulatedTimeSeries\%%y_%%z

			)
		)

		cd Evaluation
		python evaluation.py 0 0
		cd ..

		copy Evaluation\Input\EventLogs\* Results\%%x_%%a\EventLogs
		copy Evaluation\Input\PetriNets\* Results\%%x_%%a\PetriNets
		copy Evaluation\Output\Metrics\* Results\%%x_%%a\Metrics
		xcopy Evaluation\Output\TimeSeriesPairs Results\%%x_%%a\TimeSeriesPairs /E
	)
)





