# ChatGPT Research

## Setup

Use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
or [mamba](https://mamba.readthedocs.io/en/latest/installation.html).
(Environments were tested with **mamba**)

## In Miniforge:
CPU Environment:
```shell
mamba env create --file env-cpu.yml
mamba activate gpt-cpu
```

GPU Environment:
```shell
mamba env create --file env-gpu.yml
mamba activate gpt-gpu
```

## In Anaconda / Miniconda3 prompt
CPU Environment:
```shell
conda env create --file env-cpu.yml
conda activate gpt-cpu
```

GPU Environment:
```shell
conda env create --file env-gpu.yml
conda activate gpt-gpu
```

## config.py

 * `model_name` : `roberta-base`|`roberta-large` - defines model
 * `embeddings_file`: string - where embeddings should be saved
 * `ranges_file`: string - where ranges should be saved


## Run scripts

When environment is activated you can run python scripts in this project.
For example `python build_embeddings.py`.


