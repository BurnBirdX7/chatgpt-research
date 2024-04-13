from __future__ import annotations

import gc
import itertools
import logging
import os.path

import faiss  # type: ignore
from progress.bar import ChargingBar  # type: ignore
import torch  # type: ignore
import numpy as np

from transformers import RobertaTokenizer, RobertaModel  # type: ignore
from typing import List, Tuple, Optional, Callable, Dict, Any

from .pipeline.base_data_descriptor import ValueType
from .pipeline.data_descriptors import ComplexDictDescriptor
from .source_mapping import SourceMapping
from .config import EmbeddingBuilderConfig
from .pipeline import BaseNode, BaseDataDescriptor, base_data_descriptor, ListDescriptor


class EmbeddingsBuilder:
    def __init__(
        self,
        config: EmbeddingBuilderConfig,
        logger: logging.Logger = logging.getLogger(__name__),
    ) -> None:
        self.logger = logger

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = config.tokenizer
        self.model = config.model

        self.max_model_input_length = self.tokenizer.model_max_length
        self.model_output_length = self.model.config.hidden_size

        self.embedding_length = self.model.config.hidden_size
        self.normalize = config.normalize
        self.logger.debug(f"EmbeddingsBuilder: Embedding normalization: {self.normalize}")

        self.suppress_progress_report = False

        if config.centroid_file is not None:
            self.centroid = np.load(config.centroid_file)
            self.logger.debug(f'Centroid loaded from file "{config.centroid_file}"')
            self.logger.debug(f"Centroid {self.centroid}")
        else:
            self.centroid = np.zeros(self.embedding_length)
            self.logger.debug("Using default centroid")

    def from_ids(self, input_ids: List[int]) -> np.ndarray:
        """
        :param input_ids: list of token ids
        :return: numpy array with dimensions (token_count, embedding_length)
        """
        sequence_length: int = self.max_model_input_length
        window_step: int = sequence_length // 2

        embedding_len: int = self.embedding_length
        embeddings: np.ndarray = np.empty((0, embedding_len))
        previous_half: Optional[np.ndarray] = None

        window_steps = range(0, len(input_ids), window_step)
        if not self.suppress_progress_report:
            window_steps = ChargingBar("Embeddings").iter(window_steps)

        for i in window_steps:
            # Create tensor with acceptable dimensions:
            input_ids_tensor = torch.tensor(input_ids[i : i + sequence_length]).unsqueeze(0)

            # Moves tensor to model's device
            input_ids_tensor = input_ids_tensor.to(self.model.device)

            with torch.no_grad():
                output = self.model(input_ids_tensor)
            seq_embeddings = output.last_hidden_state.squeeze(0).cpu().numpy().astype(np.float32)

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

    def tensor_from_text(self, text: str) -> torch.Tensor:
        tokenizer_output = self.tokenizer(
            text,
            return_tensors="pt",
            return_attention_mask=True,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=self.max_model_input_length,
        )

        input_ids = tokenizer_output["input_ids"].reshape((-1, self.max_model_input_length)).to(self.device)
        attention_mask = tokenizer_output["attention_mask"].reshape(input_ids.shape).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids, attention_mask=attention_mask)

        return output.last_hidden_state.reshape((-1, self.model_output_length))

    def from_text(self, text: str) -> np.ndarray:
        """
        :param text: input
        :return: numpy array with dimensions (token_count, embedding_length)
                 token_count depends on text contents
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        return self.from_ids(input_ids)

    def from_sources(
        self, source_list: List[str], source_provider: Callable[[str], Dict[str, str]]
    ) -> Tuple[np.ndarray, SourceMapping]:
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
            self.logger.debug(f"Source {i + 1}/{len(source_list)} in processing")
            sources_dict = source_provider(page)

            input_ids: List[int] = []
            for title, text in sources_dict.items():
                tokens = self.tokenizer.tokenize(text)
                input_ids += self.tokenizer.convert_tokens_to_ids(tokens)
                src_map.append_interval(len(tokens), title)

            page_embeddings = self.from_ids(input_ids)
            embeddings = np.concatenate([embeddings, page_embeddings])

        return embeddings, src_map


class NDArrayDescriptor(BaseDataDescriptor[np.ndarray]):

    def store(self, data: np.ndarray) -> Dict[str, base_data_descriptor.ValueType]:
        filename = f"ndarray-{self.get_timestamp_str()}-{self.get_random_string(4)}.npy"
        path = os.path.join(self.artifacts_folder, filename)
        with open(path, "wb") as f:
            np.save(f, data)

        return {"path": os.path.abspath(path)}

    def load(self, dic: Dict[str, base_data_descriptor.ValueType]) -> np.ndarray:
        path = dic["path"]
        with open(path, "rb") as f:  # type: ignore
            return np.load(f)

    def cleanup(self, dic: dict):
        self.cleanup_files(dic["path"])

    def get_data_type(self) -> type:
        return np.ndarray


class TensorDescriptor(BaseDataDescriptor[torch.Tensor]):

    def store(self, data: torch.Tensor) -> dict[str, ValueType]:
        filename = f"tensor-{self.get_timestamp_str()}-{self.get_random_string(4)}.pt"
        path = os.path.join(self.artifacts_folder, filename)
        torch.save(data, path)
        return {"path": os.path.abspath(path)}

    def load(self, dic: dict) -> torch.Tensor:
        path = dic["path"]
        return torch.load(path)

    def cleanup(self, dic: dict):
        self.cleanup_files(dic["path"])

    def get_data_type(self) -> type[torch.Tensor]:
        return torch.Tensor

    def is_optional(self) -> bool:
        # Tensors are heavy space-wise, so their storing them is optional
        return True


class EmbeddingsFromTextNode(BaseNode):
    def __init__(self, name: str, config: EmbeddingBuilderConfig):
        super().__init__(name, [str], NDArrayDescriptor())
        self.eb_config = config

    def process(self, text: str) -> np.ndarray:
        eb = EmbeddingsBuilder(self.eb_config, logging.getLogger(f"{self.logger.name}.EmbeddingsBuilder"))
        return eb.tensor_from_text(text).cpu().numpy()

    def prerequisite_check(self) -> str | None:
        centroid_file = self.eb_config.centroid_file
        if centroid_file is not None:
            if not os.path.exists(centroid_file):
                return f'Centroid file "{self.eb_config.centroid_file}" is specified but doesn\'t exist'

        return None


class TokenizeTextNode(BaseNode):
    def __init__(self, name, config: EmbeddingBuilderConfig):
        super().__init__(name, [str], ListDescriptor())
        self.eb_config = config

    def process(self, text: str, *ignore) -> List[str]:
        tokenizer = self.eb_config.tokenizer
        tokens = tokenizer.tokenize(text)
        readable_tokens = list(map(lambda s: tokenizer.convert_tokens_to_string([s]), tokens))
        return readable_tokens


class LikelihoodsForMultipleSources(BaseNode):

    def __init__(self, name, embeddings_builder_config: EmbeddingBuilderConfig):
        super().__init__(name, [dict], ComplexDictDescriptor(TensorDescriptor()))
        self.eb_config = embeddings_builder_config

    def process(self, sources_data: Dict[str, str]) -> Dict[str, torch.Tensor]:
        tokenizer = self.eb_config.tokenizer
        model = self.eb_config.model

        source_batched_likelihoods = {}  # Dict (name -> likelihoods_batch)  batch has dimensions (batch, text, vocab)

        for i, (source_name, source_text) in enumerate(sources_data.items()):
            self.logger.debug(f'Generating likelihoods for source {i + 1}/{len(sources_data)}: "{source_name}"')
            tokenizer_output = tokenizer(
                text=source_text,
                add_special_tokens=False,
                return_tensors="pt",
                return_attention_mask=True,
                padding=True,
                pad_to_multiple_of=tokenizer.model_max_length,
            )

            def make_batch(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.reshape((-1, tokenizer.model_max_length)).to(model.device)

            source_attention_mask = make_batch(tokenizer_output["attention_mask"])
            source_token_id_batch = make_batch(tokenizer_output["input_ids"])
            with torch.no_grad():
                # Logits have dimensions: (passage, position, vocab)
                batched_logits = model(source_token_id_batch, attention_mask=source_attention_mask).logits

            batched_likelihoods = torch.nn.functional.softmax(batched_logits.cpu(), dim=2)
            source_batched_likelihoods[source_name] = batched_likelihoods

        return source_batched_likelihoods
