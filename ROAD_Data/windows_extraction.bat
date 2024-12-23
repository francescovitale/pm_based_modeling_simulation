:: %1: clustering_technique, %2 n_clusters
set clustering_technique=%1
set n_clusters=%2
set n_sensor_types=1
set sensor_types=J2

for /D %%p IN ("Output\WE\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
del /F /Q Output\WE\*

python windows_extraction.py %clustering_technique% %n_clusters% %n_sensor_types% %sensor_types%

