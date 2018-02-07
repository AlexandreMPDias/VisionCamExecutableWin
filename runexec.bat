@echo off
if NOT EXIST dist (
	echo Executable not yet created. Run pyinstaller first.
	goto:eof
)
if [%1]==[--no-log] (
	start withoutlog.bat
) else (
	start dist\\emotions\\emotions.exe > exec_log.txt 2>&1
	echo Log stored at [exec_log.txt]
	goto:eof
)
