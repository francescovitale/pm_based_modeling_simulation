del /F /Q Input\EventLogs\*
del /F /Q Input\PetriNets\*
for /D %%p IN ("Input\TimeSeries\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
del /F /Q Input\TimeSeries\*
for /D %%p IN ("Input\SimulatedTimeSeries\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
del /F /Q Input\SimulatedTimeSeries\*

del /F /Q Output\Metrics\*
for /D %%p IN ("Output\TimeSeriesPairs\*") DO (
	del /s /f /q %%p\*.*
	for /f %%f in ('dir /ad /b %%p') do rd /s /q %%p\%%f
	rmdir "%%p" /s /q
)
del /F /Q Output\TimeSeriesPairs\*

