@echo off
del /q /s temp.txt

call:stuff sad
call:stuff angry
call:stuff happy
call:stuff scared
call:stuff sunglass
call:stuff surprised
call:stuff neutral
goto: eof

:stuff
echo --%1-- >> temp.txt
ls %1 | hznt >> temp.txt
echo a | tr 'a' '\n' >> temp.txt
goto:eof