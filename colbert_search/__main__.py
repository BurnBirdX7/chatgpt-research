import subprocess
import sys
import os

colbert_rooted_scripts = ["create_index", "fever_server", "colbert_server"]
"""
These scripts must be launched with 'colbert' conda environment active 
"""


def main():
    if len(sys.argv) < 2:
        raise ValueError("Not enough parameters")

    script = sys.argv[1]

    # This file is placed in: {project_root}/colbert_search/{this_file}
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    script_path = f"{project_root}/colbert_search/{sys.argv[1]}.py"

    if script in colbert_rooted_scripts:
        module_root = os.path.join(project_root, "colbert_search/colbert_git")
        print(f'CWD for this execution is changed to "{module_root}"')
    else:
        module_root = f"{project_root}"

    args = [sys.executable, script_path, *sys.argv[2:]]
    env = os.environ.copy()
    if "PYHTONPATH" in env:
        env["PYTHONPATH"] += f":{project_root}:{module_root}"
    else:
        env["PYTHONPATH"] = f"{project_root}:{module_root}"

    print(f"PYTHONPATH: {env['PYTHONPATH']}")

    subprocess.run(args, cwd=module_root, env=env)


if __name__ == "__main__":
    main()
