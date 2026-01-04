import sys
import os
import runpy


project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


print("Starte die Anwendungs-Demo über den Projekt-Root...")
runpy.run_module('src.application_demo', run_name="__main__")