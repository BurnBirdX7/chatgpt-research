import faiss  # type: ignore
from progress.bar import ChargingBar  # type: ignore
import torch  # type: ignore
import numpy as np

from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from typing import List, Tuple, Optional, Callable, Dict

from .source_mapping import SourceMapping
from .config.EmbeddingsConfig import EmbeddingsConfig


class EmbeddingsBuilder:
    def __init__(self, config: EmbeddingsConfig):
        self.tokenizer = config.tokenizer
        self.model = config.model
        self.max_sequence_length = self.model.config.max_position_embeddings
        self.embedding_length = self.model.config.hidden_size
        self.normalize = config.normalize
        print(f"Embedding normalization: {self.normalize}")

        self.suppress_progress = False

        if config.centroid_file is not None:
            self.centroid = np.load(config.centroid_file)
            print('Centroid loaded!')
        else:
            self.centroid = np.zeros(self.embedding_length)
            print('No centroid')

    def from_ids(self, input_ids: List[int]) -> np.ndarray:
        """
        :param input_ids: list of token ids
        :return: numpy array with dimensions (token_count, embedding_length)
        """
        sequence_length: int = self.max_sequence_length
        shorten_by: int = 2 if sequence_length % 2 == 0 else 3
        sequence_length -= shorten_by
        window_step: int = sequence_length // 2

        embedding_len: int = self.embedding_length
        embeddings: np.ndarray = np.empty((0, embedding_len))
        previous_half: Optional[np.ndarray] = None

        iterable = range(0, len(input_ids), window_step)
        if not self.suppress_progress:
            iterable = ChargingBar('Embeddings').iter(iterable)

        for i in iterable:
            # Create tensor with acceptable dimensions:
            input_ids_tensor = torch.tensor(input_ids[i: i + sequence_length]).unsqueeze(0)

            # Moves tensor to model's device
            input_ids_tensor = input_ids_tensor.to(self.model.device)

            output = self.model(input_ids_tensor)
            seq_embeddings = output.last_hidden_state.detach().squeeze(0).cpu().numpy().astype(np.float32)

            if previous_half is not None:
                # Get mean value of 2 halves (prev[:t] and curr[t:])
                current_half = (previous_half + seq_embeddings[:window_step]) / 2
                embeddings = np.concatenate([embeddings, current_half])
            else:
                embeddings = seq_embeddings[:window_step]

            previous_half = seq_embeddings[window_step:]

        if previous_half is not None:
            embeddings = np.concatenate([embeddings, previous_half])

        count, length = embeddings.shape
        assert count == len(input_ids)
        assert length == embedding_len

        embeddings -= self.centroid
        if self.normalize:
            faiss.normalize_L2(embeddings)

        return embeddings

    def from_text(self, text: str) -> np.ndarray:
        """
        :param text: input
        :return: numpy array with dimensions (token_count, embedding_length)
                 token_count depends on text contents
        """

        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return self.from_ids(input_ids)

    def from_sources(self, source_list: List[str], source_provider: Callable[[str], Dict[str, str]])\
            -> Tuple[np.ndarray, SourceMapping]:
        """
        Computes embeddings for online Wikipedia
        Target pages are specified via config variable `page_names`
        :param source_list: Name of the source to be passed to source_provider
        :param source_provider: Function that accepts source name and
                                returns dictionary sub_source -> source_text
        :return: Tuple:
                    - Embeddings as 2d numpy.array
                    - and Interval to Source mapping
        """
        src_map = SourceMapping()
        embeddings = np.empty((0, self.model.config.hidden_size))

        for i, page in enumerate(source_list):
            print(f"Source {i + 1}/{len(source_list)} in processing")
            sources_dict = source_provider(page)

            input_ids: List[int] = []
            for title, text in sources_dict.items():
                tokens = self.tokenizer.tokenize(text)
                input_ids += self.tokenizer.convert_tokens_to_ids(tokens)
                src_map.append_interval(len(tokens), title)

            page_embeddings = self.from_ids(input_ids)
            embeddings = np.concatenate([embeddings, page_embeddings])

        return embeddings, src_map
