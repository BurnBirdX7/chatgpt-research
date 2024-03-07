# ColBERT index

## Initialization

Download ColBERT, ColVERTv2 checkpoint and FEVER dataset by executing `./install_colbert.sh`

Use one of the environments provided by ColBERT to run code in this module


```shell
# From project root
cd colbert_search/colbert/
conda env create --file conda_env.yml
```

You may need to install pip requirements by hand, in my case I had to run this:
```shell
conda activate colbert
python -m pip install -r condaenv.pp1wnr2r.requirements.txt
```

The requirements file was generated during conda environment creation, but wasn't installed for some reason.


When using GPU environment, you install CUDA Toolkit of specific version:
executing `conda list pytorch` (with colbert environment activated) will help you determine which one you need.
[**[CUDA Toolkit Achieve](https://developer.nvidia.com/cuda-toolkit-archive)**].
