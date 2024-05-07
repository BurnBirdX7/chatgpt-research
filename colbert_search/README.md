# ColBERT Search

## Init

Before running anything from this package:

1. Download ColBERT, ColVERTv2 checkpoint and FEVER dataset by executing:
    ```shell
    # From project root
    ./colbert_search/install_colbert.sh
    ```

2. Use Conda environment provided in file `colbert_search/colbert_env_gpu.yml`
    ```shell
    # From project root
    conda env create --file colbert_search/colbert_env_gpu.yml
    ```

## Run

### Server

```shell
python -m colbert_search colbert_server
```

Set `COLBERTSERVERCONFIG_PORT` environment variable to configure port

### Data processing

#### Indexes

To prepare wiki dumps as ColBERT indexes:

```shell
python -m colbert_search prepare_wiki
```

See `python -m colbert_search prepare_wiki --help`.

#### FEVER

For FEVER-based tests, FEVER articles can be added to the indexes

`python -m colbert_search fever_collect_dataset`



