@echo off
setlocal
if "%SUMO_HOME%"=="" (
  echo SUMO_HOME ist nicht gesetzt.
  pause
  exit /b 1
)
cd /d "%~dp0"
netconvert -n network\constructed.nod.xml -e network\constructed.edg.xml -o network\constructed.net.xml
if errorlevel 1 (
  echo Fehler beim Erzeugen des SUMO-Netzes.
  pause
  exit /b 1
)
sumo-gui -c config\constructed.sumocfg
