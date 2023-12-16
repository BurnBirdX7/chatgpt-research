import subprocess
import sys
import os

def main():
    if len(sys.argv) < 2:
        raise ValueError("Not enough parameters")

    project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    colbert_root = f"{project_root}/colbert"
    script = f"{project_root}/colbert/{sys.argv[1]}.py"

    args = [sys.executable, script, *sys.argv[2:]]
    env = os.environ.copy()
    if 'PYHTONPATH' in env:
        env['PYTHONPATH'] += f";{colbert_root}"
    else:
        env['PYTHONPATH'] = colbert_root

    subprocess.run(args, cwd=colbert_root, env=env)


if __name__ == '__main__':
    main()
