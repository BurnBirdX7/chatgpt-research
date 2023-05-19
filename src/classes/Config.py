import os.path
from typing import Dict, Any


class ConfigLoader(type):
    def __new__(cls, clsname, bases, attrs, config_file: str):
        lc: Dict[str, Any] = {}
        gl = {}
        with open(config_file, 'r') as f:
            exec(f.read(), gl, lc)

        artifacts: str = lc['artifacts_folder'] if 'artifacts_folder' in lc else 'artifacts'

        if not os.path.exists(artifacts):
            os.makedirs(artifacts)

        for name, val in lc.items():
            if type(val) == str and ('path' in name or 'file' in name):
                val = os.path.join(artifacts, val)
            attrs[name] = val

        return super().__new__(cls, clsname, bases, attrs)


class Config(metaclass=ConfigLoader, config_file='config.py'):
    pass
