import os
import subprocess
import sys


def main():
    if len(sys.argv) < 2:
        raise ValueError("Not enough parameters")

    script = f"{sys.argv[1]}.py"
    project_root = os.path.join(os.path.dirname(__file__), "colbert")

    args = [sys.executable, script, *sys.argv[2:]]

    env = os.environ.copy()
    if 'PYHTONPATH' in env:
        env['PYTHONPATH'] += f";{project_root}"
    else:
        env['PYTHONPATH'] = project_root

    subprocess.run(args, cwd=project_root, env=env)


if __name__ == '__main__':
    main()
