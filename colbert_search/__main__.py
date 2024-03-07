import subprocess
import sys
import os

colbert_rooted_scripts = ['create_index', 'fever_server']

def main():
    if len(sys.argv) < 2:
        raise ValueError("Not enough parameters")

    script = sys.argv[1]
    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    if script in colbert_rooted_scripts:
        module_root = f"{project_root}/colbert_search/colbert"
    else:
        module_root = f"{project_root}"
    script_path = f"{project_root}/colbert_search/{sys.argv[1]}.py"

    args = [sys.executable, script_path, *sys.argv[2:]]
    env = os.environ.copy()
    if 'PYHTONPATH' in env:
        env['PYTHONPATH'] += f";{module_root}"
    else:
        env['PYTHONPATH'] = module_root

    subprocess.run(args, cwd=module_root, env=env)


if __name__ == '__main__':
    main()
