from typing import Union, Tuple, List
from transformers import RobertaTokenizer, RobertaModel  # type: ignore
import pandas as pd  # type: ignore
import faiss  # type: ignore
import numpy as np


from IntervalToSource import IntervalToSource
from text_embedding import input_ids_embedding, text_embedding
from wiki import parse_wiki
import roberta
import config


def build_embeddings_from_wiki(tokenizer: RobertaTokenizer, model: RobertaModel) -> Tuple[np.ndarray, IntervalToSource]:
    """
    Computes embeddings
    :param tokenizer: Tokenizer instance
    :param model: Model instance
    :return: Tuple:
                - Embeddings as 2d numpy.array
                - and Interval to Source mapping
    """
    src_map = IntervalToSource()
    embeddings = np.empty((0, model.config.hidden_size))
    page_names = config.page_names

    for i, page in enumerate(page_names):
        print(f"Source {i + 1}/{len(page_names)} in processing")
        sections_dict = parse_wiki(page)

        input_ids: List[int] = []
        for title, text in sections_dict.items():
            tokens = tokenizer.tokenize(text)
            input_ids += tokenizer.convert_tokens_to_ids(tokens)
            src_map.append_interval(len(tokens), title)

        page_embeddings = input_ids_embedding(input_ids, model)
        embeddings = np.concatenate([embeddings, page_embeddings])

    return embeddings, src_map


def build_index_from_embeddings(embeddings: Union[np.ndarray, pd.DataFrame], use_gpu: bool = config.faiss_use_gpu) -> faiss.Index:
    """
    Builds index from provided embeddings
    :param embeddings: data to build the index
    :param use_gpu: if set, GPU is used to build the index
    :return: IndexFlatIP, or GpuIndexFlatIP id use_gpu is True
    """
    # C-contiguous order and np.float32 type are required
    if isinstance(embeddings, np.ndarray) and embeddings.flags['C_CONTIGUOUS']:
        data = embeddings.astype(np.float32)
    else:
        data = np.array(embeddings, order="C", dtype=np.float32)

    sequence_len, embedding_len = data.shape

    faiss.normalize_L2(data)
    print("Building index... ", end="")
    index = faiss.IndexFlatIP(embedding_len)
    if use_gpu:
        gpu_res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(gpu_res, 0, index)

    index.add(data)
    print("Done")
    return index


def build_index() -> Tuple[faiss.Index, IntervalToSource]:
    """
    :returns: faiss index and interval to source mapping
    """
    tokenizer, model = roberta.get_default()
    e, r = build_embeddings_from_wiki(tokenizer, model)
    index = build_index_from_embeddings(e, False)
    return index, r


def main() -> None:
    index, mapping = build_index()
    faiss.write_index(index, config.index_file)
    mapping.to_csv(config.ranges_file)


if __name__ == '__main__':
    main()
