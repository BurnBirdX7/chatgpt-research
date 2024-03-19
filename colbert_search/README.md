# ColBERT index

## Initialization

Download ColBERT, ColVERTv2 checkpoint and FEVER dataset by executing:
```shell
# From project root 
cd colbert_search
./install_colbert.sh
```

Use Conda environment provided in file `colbert_search/colbert_env_gpu.yml`

```shell
# From project root
conda env create --file colbert_search/colbert_env_gpu.yml
```

When using GPU environment, you install CUDA Toolkit of specific version:
executing `conda list pytorch` (with colbert environment activated) will help you determine which one you need.
[**[CUDA Toolkit Achieve](https://developer.nvidia.com/cuda-toolkit-archive)**].
