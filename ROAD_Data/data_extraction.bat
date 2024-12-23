set n_sensor_types=1
set sensor_types=J2

for /D %%p IN ("Output\DE\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
del /F /Q Output\DE\*

python data_extraction.py %n_sensor_types% %sensor_types%

