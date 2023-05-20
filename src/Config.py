import os.path
from typing import Dict, Any, List


class ConfigLoader(type):
    def __new__(cls, clsname, bases, attrs, config_file: str):
        lc: Dict[str, Any] = {}
        gl: Dict[str, Any] = {}
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
    # Misc
    model_name: str
    faiss_use_gpu: bool
    show_plot: bool
    threshold: float

    # Files
    artifacts_folder: str
    embeddings_file: str
    mapping_file: str
    index_file: str
    centroid_file: str

    # Wiki Articles
    page_names: List[str]
    unrelated_page_names: List[str]
    unrelated_page_names_2: List[str]
