# Fact-checking for LLM's Output 

## Setup

[conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) is required.

Provided conda environments:
 * CPU-only:
   * `factcheck-env-cpu.yml`
   * `factcheck-env-cpu-dev.yml`
 * CUDA:
   * `factcheck-env.yml`
   * `factcheck-env-dev.yml`

`-dev` environments contain additional packages for testing and maintaining the code.

Import and activate chosen environment:
```shell
conda env create --file <ENV_FILE>
conda activate factcheck
```

## Modules

* `src` - resusable code
* `scripts` - scripts created to do something specific, although functions can be reused
* `colbert_search` - scripts focused around ColBERT, require different environment
  * provides `colbert_server`
    * To use it, you need to create index of all sources you want to consider during the analysis
    * `colbert_search.wiki_pipeline` is a script that can create ColBERT indexes from wikipedia dumps
      (multistream divided into multiple archives)
* `server` - web server that provides GUI for possible sources search, requires running `colbert_server` to run
* `test` - unit tests for some features, use `pytest` to run them
* `search` - old, unmaintained search, that was preforming the same role as `colbert_search`
  but based on Lucene full-text search.
   * [PyLucene](#pylucene) is required to use it,
  it has no integration in the current process, and also you'd need to compile Lucene index
  of all the sources you want to consider a possibility in the analysis 



## Run scripts

All tasks are a part of `scripts` or `colbert_search` module, and shouldn't be run directly.

When environment is activated you can run python scripts in this project.
To run scripts use:
```shell
# in project root
python -m <module-name> <script-name> [scipt-args...]
```

### Scripts

Scripts aren't systematized, you'd need to check docstring inside each one to understand what they are doing.

Most important scripts are:
 * `scripts` module:
   * `coloring_pipeline` - is not a script,
     it provides a single function - `get_coloring_pipeline` that currently represents all the analysis
   * `estimate_centroid` - text embeddings are often skewed from 0 coordinates, compute centroid to offset them back,
      and to improve cosine distance comparison quality.
   * `estimate_thresholds` - computes cosine distance thresholds that we can use to reject unrelated 
     embeddings when performing knn
 * `colbert_search`
   * `wiki_pipeline` is a script that can create ColBERT indexes from wikipedia dumps
     (multistream divided into multiple archives)
   * `colbert_server` server that provides ColBERT full-text search on-demand and on multiple ColBERT Indexes
   * `prepare_wiki` script that converts wikitext into ColBERT-readable passages, part of `wiki_pipeline`

## PyLucene

[!] _Unmaintained section_

To run `build_index_from_potential_sources` script you need [PyLucene](https://lucene.apache.org/pylucene/) installed in the environment.
To do it you need to build PyLucene.

### Linux / WSL

Requirements:
 * **setuptools**: `conda install setuptools`
 * (_optional, recommended_) **Temurin JDK 17**: [Instruction](https://adoptium.net/installation/linux/)

Build:\
_Instruction is close to [this one](https://lucene.apache.org/pylucene/install.html), but changed a little_
* Download [PyLucene 9.6.0](https://dlcdn.apache.org/lucene/pylucene/pylucene-9.6.0-src.tar.gz)
  * Unpack and `cd` into unpacked `pylucene-9.6.0` directory
* Activate conda environment:
  ```shell
  conda activate gpt-gpu
  ```
* Build JCC:
  * If you didn't install **Temurin JDK 17**, set environment variable `JCC_JDK` to Java's home directory (one that contains `bin` dir)
  * ```shell
    pushd jcc
    python setup.py build
    python setup.py install  # without sudo
    popd
    ```
* Edit `Makefile` to match your system. You'll need to uncomment one section and edit paths. More detailed instructions are inside the file.
  * Specify path to python **of active environmen**t, not to system python
  * You can get the path by calling `whereis python` when environment is active
* Build: `make`
* Run tests: `make test`, should be no errors
* Install: `make install` (**without sudo**). 
