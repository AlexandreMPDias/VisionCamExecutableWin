@echo off
logger.bat %1
if EXIST dist (
	if not exist dist/emotion echo Build failed.
) else (
	if [%1] == [--no-log] (
		runexec.bat --no-log
	) else (
		runexec.bat
	)
)