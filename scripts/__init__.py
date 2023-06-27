import os

this_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(this_path)
project_dir = os.path.dirname(scripts_dir)

cwd = os.getcwd()

if cwd != project_dir:
    print(f"[scripts] Changed working directory from {cwd} to {project_dir}")
    os.chdir(project_dir)