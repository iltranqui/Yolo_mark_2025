@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=amd64 -host_arch=amd64 -no_logo
MSBuild.exe yolo_mark.sln /p:Configuration=Release /p:Platform=x64 /p:PlatformToolset=v143
