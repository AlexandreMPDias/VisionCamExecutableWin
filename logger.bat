@echo off
call:check_and_delete dist
call:check_and_delete build

echo [ Building ]
call installer.bat > build_log.txt 2>&1
if not exist dist (
	echo [ Build failed ]
	goto:eof
)
if not exist dist\\emotions (
	echo [ Build failed ]
	goto:eof
)
echo [ Building Completed ]
echo a | tr 'a' '\n'
call:countmsg Warning WARNING build_log.txt
call:countmsg Error ERROR build_log.txt

if NOT [%1] == [] cat build_log.txt
goto:eof

:: Checks if a folder exists, if it does, deletes it.
::			<folder_name>
:check_and_delete 
if EXIST %1 (
	rmdir /s /q %1
	if EXIST $1 (
		echo [%1] unable to delete directory.
	) else (
		echo [%1] deleted.
	)
) else (
	echo [%1] was already deleted.
)
goto :eof

:: Counts Messages in the log
:: 	<message> <upper> <location>
:countmsg
setlocal
set "msg=%1"
set "upper=%2"
set "location=%3"
echo %msg%s found: | tr -d "\r\n"
findstr /R /N "^.*%upper%.*$" %location% | find /c ":"
REM ::echo [ findstr /R /N "^.*INFO.*$" %location% | find /c ":" ] %msg% found.
goto:eof

:: echo <mssg>
:echo2
echo %1 | tr -d "\r\n"
goto:eof