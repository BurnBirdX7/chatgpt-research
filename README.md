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

### Misc
 * `model_name` : `roberta-base`|`roberta-large` - defines model
 * `faiss_use_gpu`: bool - use GPU when building FAISS index or not
 * `show_plot`: bool - show plot after estimation script run
 * `threshold`: float - Lowest acceptable cosine distance, when comparing embeddings

### Files
 * `artifacts_folder`: string - folder that contains all the artifacts -
                         files that can be loaded by scripts, and files produced by the scripts
 * `embeddings_file`: string - where embeddings should be saved _\[Not Used\]_
 * `mapping_file`: string - where mapping of embedding index to source is saved (CSV)
 * `index_file`: string - where FAISS index is saved (faiss format)
 * `centroid_file`: sting - where embeddings centroid is saved (Numpy's NPY file)

### Wiki Articles
 * `page_names`: list\[string\] - list of Wikipedia articles on _target topic_
 * `unrelated_page_names`: list\[string\] - list of Wikipedia articles not on _target topic_,
   * used when estimating thresholds and centroid
 * `unrelated_page_names_2`: list\[string\] - large list of Wikipedia articles not on _target topic_,
   * used for centroid estimation

## Run scripts

All runnable scripts placed in `scripts` folder.

When environment is activated you can run python scripts in this project.
For example `python scriptbuild_embeddings.py`

### Scripts

and their vague description...

 * `build_index.py` - builds embedding index from wiki articles
 * `collect_pop_quiz.py` - surveys ChatGPT for answers on quiz
   * Quiz name should be supplied as first parameter `python scripts/collect_pop_quiz.py test_quiz`
   * See details in top comment in the file
   * Requires environment variable `OPENAI_API_KEY` to be set
 * `estimate_centroid.py` - collects large amount of embeddings from wiki articles and computes centroid estimation
 * `estimate_thresholds.py` - collects data on related and unrelated topics and estimates threshold
 * `filter_answers.py` - filters correct answers provided by ChatGPT
   * See details in top comment in the file
 * `format_questions.py` - Formats quiz _\[ Deprecated \]_
 * `model_of_GPT.py` - ??? Ask Misha
 * `print_answers.py` - Prints incorrect answers given by ChatGPT _\[ Deprecated \]_
 * `survey.py`
   * See details in top comment in the file


