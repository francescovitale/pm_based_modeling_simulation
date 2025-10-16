:: Options:
:: anomaly_type=[V, W]; one anomaly type at a time
:: pd_variant=[im, imf25, imf50, imf75, imf99, ilp, ilp25, ilp50, ilp75, ilp99, alpha]
:: n_clusters=<integer>
:: n_simulation_traces=<integer>
:: clustering_technique=[kmeans, agglomerative, gmm]
:: simulation_technique=[process_mining]

set pd_variant=imf75 ilp75 hm
set n_clusters=2 3 4 5 6 7 8 9 10 11 12
set n_simulation_traces=300
set clustering_technique=kmeans
set n_reps=7

for /D %%p IN ("Results\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)

for /D %%p IN ("ResultsNormal\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)


:: for n_reps times:
:: 		ROAD data windows are generated
:: 		for each anomaly type:
:: 			the framework is executed

for /l %%a in (1, 1, %n_reps%) do (

	mkdir Results\%%a
	mkdir ResultsNormal\%%a
	for %%y in (%n_clusters%) do (
	
		cd ROAD_Data
		call windows_extraction %clustering_technique% %%y
		cd ..
	
		for %%z in (%pd_variant%) do (
			mkdir Results\%%a\V_%%y_%%z\TimeSeries
			mkdir Results\%%a\V_%%y_%%z\TestEventLog
			mkdir Results\%%a\V_%%y_%%z\PetriNet
			mkdir Results\%%a\V_%%y_%%z\SimulatedTimeSeries
			
			mkdir Results\%%a\W_%%y_%%z\TimeSeries
			mkdir Results\%%a\W_%%y_%%z\TestEventLog
			mkdir Results\%%a\W_%%y_%%z\PetriNet
			mkdir Results\%%a\W_%%y_%%z\SimulatedTimeSeries
			
			mkdir ResultsNormal\%%a\V_%%y_%%z\TimeSeries
			mkdir ResultsNormal\%%a\V_%%y_%%z\PetriNet
			mkdir ResultsNormal\%%a\V_%%y_%%z\SimulatedTimeSeries
			
			mkdir ResultsNormal\%%a\W_%%y_%%z\TimeSeries
			mkdir ResultsNormal\%%a\W_%%y_%%z\PetriNet
			mkdir ResultsNormal\%%a\W_%%y_%%z\SimulatedTimeSeries
			
			copy ROAD_Data\Output\WE\Test\V\* Results\%%a\V_%%y_%%z\TimeSeries
			copy ROAD_Data\Output\WE\Test\W\* Results\%%a\W_%%y_%%z\TimeSeries
			copy ROAD_Data\Output\WE\Test\V\* ResultsNormal\%%a\V_%%y_%%z\TimeSeries
			copy ROAD_Data\Output\WE\Test\W\* ResultsNormal\%%a\W_%%y_%%z\TimeSeries
		
			call framework_case_study %%z 1.0 V
			
			copy Output\ELE\EventLog\EL_TST.xes Results\%%a\V_%%y_%%z\TestEventLog
			copy Output\PD\PetriNet\* Results\%%a\V_%%y_%%z\PetriNet
			xcopy Output\S\Data Results\%%a\V_%%y_%%z\SimulatedTimeSeries /E
			
			call framework_case_study %%z 1.0 W
			
			copy Output\ELE\EventLog\EL_TST.xes Results\%%a\W_%%y_%%z\TestEventLog
			copy Output\PD\PetriNet\* Results\%%a\W_%%y_%%z\PetriNet
			xcopy Output\S\Data Results\%%a\W_%%y_%%z\SimulatedTimeSeries /E
			
			
			call framework_case_study_normal %%z V
			
			copy Output\PD\PetriNet\* ResultsNormal\%%a\V_%%y_%%z\PetriNet
			xcopy Output\S\Data ResultsNormal\%%a\V_%%y_%%z\SimulatedTimeSeries /E
			
			call framework_case_study_normal %%z W
			
			copy Output\PD\PetriNet\* ResultsNormal\%%a\W_%%y_%%z\PetriNet
			xcopy Output\S\Data ResultsNormal\%%a\W_%%y_%%z\SimulatedTimeSeries /E
			
			
		)
	)
)





