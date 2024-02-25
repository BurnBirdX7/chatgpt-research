# ChatGPT Research

## Setup

Use [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

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

Or the same commands using `conda`.

## PyLucene
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

## config.py

### Used variables
#### Misc
 * `model_name` : `roberta-base`|`roberta-large` - defines model
 * `faiss_use_gpu`: bool - use GPU when building FAISS index or not
 * `show_plot`: bool - show plot after estimation script run
 * `threshold`: float - Lowest acceptable cosine distance, when comparing embeddings

#### Files
 * `artifacts_folder`: string - folder that contains all the artifacts -
                         files that can be loaded by scripts, and files produced by the scripts
 * `embeddings_file`: string - where embeddings should be saved _\[Not Used\]_
 * `mapping_file`: string - where mapping of embedding index to source is saved (CSV)
 * `index_file`: string - where FAISS index is saved (faiss format)
 * `centroid_file`: sting - where embeddings centroid is saved (Numpy's NPY file)
 * `temp_index_file`: string - where temporary index built from query text is written
 * `source_index_path`: string - where Lucene source index is located 

#### Wiki Articles
 * `page_names`: list\[string\] - list of Wikipedia articles on _target topic_
 * `unrelated_page_names`: list\[string\] - list of Wikipedia articles not on _target topic_,
   * used when estimating thresholds and centroid
 * `unrelated_page_names_2`: list\[string\] - large list of Wikipedia articles not on _target topic_,
   * used for centroid estimation

### Loading from environment

You can set variable as `None` and in this case Config will try to load it from environment.
For example `source_index_path` is set to None and value will be acquired from `SOURCE_INDEX_PATH` environment variable.

### Paths and files

Variable that has `file` or `path` at the end of the name:
 * Must contain path
 * If it's not an absolute path, it must be a path relative to **artifacts folder**

## Run scripts

All runnable scripts placed in `scripts` folder.

When environment is activated you can run python scripts in this project.
To run scripts use:
```shell
# in project root
python -m scripts <script-name> [scipt-args...]
```

### Scripts

and their vague description...

 * `build_index` - builds embedding index from wiki articles
 * `collect_pop_quiz` - surveys ChatGPT for answers on quiz
   * Quiz name should be supplied as first parameter `python -m scripts collect_pop_quiz test_quiz`
   * See details in top comment in the file
   * Requires environment variable `OPENAI_API_KEY` to be set
 * `estimate_centroid` - collects large amount of embeddings from wiki articles and computes centroid estimation
 * `estimate_thresholds` - collects data on related and unrelated topics and estimates threshold
 * `filter_answers` - filters correct answers provided by ChatGPT
   * See details in top comment in the file
 * `format_questions` - Formats quiz _\[ Deprecated \]_
 * `model_of_GPT` - ??? Ask Misha
 * `print_answers` - Prints incorrect answers given by ChatGPT _\[ Deprecated \]_
 * `survey`
   * See details in top comment in the file


 ## Architecture
 ### Architecture in general
![arch](https://github.com/BurnBirdX7/chatgpt-research/assets/55112338/068cca63-25e1-45f2-b0d0-0b3a87449ff3)
 ### Sequence diagram
![ARCH_GPT](https://github.com/BurnBirdX7/chatgpt-research/assets/55112338/a2f35b2e-b925-4837-9cf2-0c23c4b8aa11)


## Buisness domain and subdomains
Business Domain: сервис/инструмент позволяющий получить первоначальные
источники информации, которая включена в ответ gpt. Абсолютно новый подход
в данной сфере. Например по полученным ссылкам можно судить насколько
правдив ответ gpt, насколько мы можем ему доверять. Можем также узнать
дополнительную информацию из первоначальных источников, которая поможет
сделать/изменить новый вывод о полученном ответе. Данный инструмент может
выступать в роли fact checking, также опираясь и анализируя ресурсы, делать
вывод.
### Core subdomains:
- раскраска (это новый разработанный подход для определения источников,
агрегированию информации и финальному присвоению ссылок на источники).
### Generic subdomains:
- инструмент для хранение информации из источников (потому что используем
готовый open source инструмент);
- языковая модель (потому что используем готовую open source модель);
- инструмент для поиска (потому что используем готовый open source
инструмент);
- инструмент для полнотекстового поиска (потому что используем готовый open
source инструмент);
- сервис аутентификации (потому что используем готовый open source сервис);
- сервис оплаты (потому что используем готовый open source сервис).
### Supporting subdomains:
- web-ui интерфейс (так как разработать ui интерфейс не составляет очень
большого труда).

## Context map
  ![BC](https://github.com/BurnBirdX7/chatgpt-research/assets/55112338/242c69de-fbbd-4d51-94a6-dcce403b1806)

## Layered architecture 
 ![LA](https://github.com/BurnBirdX7/chatgpt-research/assets/55112338/b248b669-9e7d-4e85-bb82-ae6e01877dbe)
