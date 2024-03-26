from __future__ import annotations

import copy
import datetime
import time

import numpy as np
import torch
from typing import Optional, List, Set, Any, Dict

from src.config import EmbeddingBuilderConfig
from src.pipeline import BaseNode, BaseDataDescriptor, base_data_descriptor, DictDescriptor


class Chain:
    begin_pos: int
    end_pos: int
    likelihoods: List[float]
    skips: int = 0
    source: str

    def __init__(self, source: str,
                 begin_pos: int = None,   # type: ignore
                 end_pos: int = None,     # type: ignore
                 likelihoods: Optional[List[float]] = None,
                 skips: int = 0):
        self.begin_pos: int = begin_pos
        self.end_pos: int = end_pos
        self.likelihoods = [] if (likelihoods is None) else likelihoods
        self.source = source
        self.skips = skips

    def __len__(self) -> int:
        if self.begin_pos is None:
            return 0

        return self.end_pos - self.begin_pos + 1

    def __str__(self) -> str:
        return (f"Chain {{\n"
                f"\tseq = {self.begin_pos}..{self.end_pos}\n"
                f"\tlikelihoods = {self.likelihoods}\n"
                f"\tskips = {self.skips}\n"
                f"\tscore = {self.get_score()}\n"
                f"\tsource = {self.source}\n"
                f"}}\n")

    def __repr__(self) -> str:
        return (f"Chain("
                f"begin_pos={self.begin_pos}, "
                f"end_pos={self.end_pos}, "
                f"likelihoods={self.likelihoods!r}, "
                f"source={self.source!r}, "
                f"skips={self.skips}"
                f")")

    def to_dict(self) -> dict:
        return {
            "begin_pos": self.begin_pos,
            "end_pos": self.end_pos,
            "likelihoods": self.likelihoods,
            "skips": self.skips,
            "source": self.source
        }

    @staticmethod
    def from_dict(d: dict) -> "Chain":
        return Chain(
            begin_pos=d["begin_pos"],
            end_pos=d["end_pos"],
            likelihoods=d["likelihoods"],
            skips=d["skips"],
            source=d["source"]
        )

    def append(self, likelihood: float, position: int) -> None:
        self.likelihoods.append(likelihood)
        if self.begin_pos is None:
            self.begin_pos = position
            self.end_pos = position
        else:
            if self.end_pos + self.skips + 1 != position:
                raise ValueError(f"{self.end_pos=}, {position=}")
            self.end_pos += self.skips + 1
        self.skips = 0

    def skip(self) -> None:
        self.skips += 1

    def get_token_positions(self) -> Set[int]:
        return set(range(self.begin_pos, self.end_pos + 1))

    def get_score(self):
        # log2(2 + len) * ((lik_h_0 * ... * lik_h_len) ^ 1 / len)   = score
        l = np.exp(np.log(self.likelihoods).mean())
        score = l * len(self.likelihoods)**2
        return score

    @staticmethod
    def generate_chains(source_len: int, likelihoods: torch.Tensor,
                        token_ids: List[int], token_start_pos: int, source_name: str) -> List[Chain]:
        """
        Generates chains of tokens with the same source
        """
        result_chains: List[Chain] = []

        for source_start_pos in range(0, source_len):
            chain = Chain(source_name)
            shift_upper_bound = min(source_len - source_start_pos, len(token_ids) - token_start_pos)
            for shift in range(0, shift_upper_bound):
                token_pos = token_start_pos + shift
                source_pos = source_start_pos + shift

                assert token_pos < len(token_ids)
                assert source_pos < source_len

                token_curr_id = token_ids[token_pos]
                token_curr_likelihood = likelihoods[source_pos][token_curr_id].item()

                if token_curr_likelihood < 1e-5:
                    chain.skip()
                    if chain.skips > 3:
                        break
                else:
                    chain.append(token_curr_likelihood, token_pos)
                    if len(chain) > 1:
                        result_chains.append(copy.deepcopy(chain))

        return result_chains


class ChainListDescriptor(BaseDataDescriptor):

    def store(self, data: List[Chain]) -> dict[str, base_data_descriptor.ValueType]:
        return {
            "chains": [
                chain.to_dict()
                for chain in data
            ]
        }

    def load(self, dic: dict[str, base_data_descriptor.ValueType]) -> List[Chain]:
        return [
            Chain.from_dict(d)      # type: ignore
            for d in dic["chains"]  # type: ignore
        ]

    def get_data_type(self) -> type[list]:
        return list


class ChainingNode(BaseNode):
    def __init__(self, name: str, embedding_builder_config: EmbeddingBuilderConfig):
        super().__init__(name, [str, list, dict], ChainListDescriptor())
        self.eb_config = embedding_builder_config

    def process(self, input_text: str, sources: List[str], sources_data: Dict[str, str]) -> Any:
        tokenizer = self.eb_config.tokenizer
        model = self.eb_config.model

        model_time = 0
        chaining_time = 0

        input_token_ids = tokenizer.encode(input_text)

        source_batched_likelihoods = {}  # Dict (name -> likelihoods_batch)  batch has dimensions (batch, text, vocab)
        unique_sources = set(sources)
        for i, source in enumerate(unique_sources):
            print(f"Producing embeddings for source {i+1}/{len(unique_sources)}: \"{source}\"")
            source_text = sources_data[source]
            tokenizer_output = tokenizer(text=source_text, add_special_tokens=False,
                                         return_tensors='pt', return_attention_mask=True,
                                         padding=True, pad_to_multiple_of=tokenizer.model_max_length)

            def make_batch(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.reshape((-1, tokenizer.model_max_length)).to(model.device)

            source_attention_mask = make_batch(tokenizer_output['attention_mask'])
            source_token_id_batch = make_batch(tokenizer_output['input_ids'])
            model_start_time = time.time()
            with torch.no_grad():
                # Logits have dimensions: (passage, position, vocab)
                batched_logits = model(source_token_id_batch, attention_mask=source_attention_mask).logits
            batched_likelihoods = torch.nn.functional.softmax(batched_logits, dim=2)
            model_time += time.time() - model_start_time
            source_batched_likelihoods[source] = batched_likelihoods.cpu()

        result_chains = []
        for token_pos, (token_id, source) in enumerate(zip(input_token_ids, sources)):
            print(f"position: {token_pos + 1} / {len(input_token_ids)}")

            batched_likelihoods = source_batched_likelihoods[source]
            print(f"batch size: {len(batched_likelihoods)}, "
                  f"token id: {token_id}, "
                  f"source: {source}")

            chaining_start_time = time.time()
            for i, passage_likelihoods in enumerate(batched_likelihoods):
                print(f"\tpassage #{i+1}")
                result_chains += Chain.generate_chains(len(passage_likelihoods), passage_likelihoods, input_token_ids, token_pos,
                                                       source)
            chaining_time += time.time() - chaining_start_time

        print(f"[STAT] > model time: {datetime.timedelta(seconds=model_time)}")
        print(f"[STAT] > chaining time: {datetime.timedelta(seconds=chaining_time)}")

        torch.cuda.empty_cache()
        return result_chains


class FilterChainsNode(BaseNode):
    """
    Removes intersections between chains giving priority to chains with higher score
    """

    def __init__(self, name: str):
        super().__init__(name, [list], ChainListDescriptor())

    def process(self, chains: List[Chain]) -> List[Chain]:
        filtered_chains: List[Chain] = []
        marked_positions: Set[int] = set()  # positions that are marked with some source
        for chain in sorted(chains, key=lambda x: x.get_score(), reverse=True):
            positions = chain.get_token_positions()
            marked_positions_inside_chain = marked_positions.intersection(positions)
            if len(marked_positions_inside_chain) == 0:
                marked_positions |= positions
                filtered_chains.append(chain)

        return filtered_chains


class Pos2ChainMappingDescriptor(BaseDataDescriptor[Dict[int, Chain]]):

    def store(self, data: Dict[int, Chain]) -> dict[str, base_data_descriptor.ValueType]:
        return {
            str(pos): chain.to_dict()
            for pos, chain in data.items()
        }

    def load(self, dic: dict[str, base_data_descriptor.ValueType]) -> Dict[int, Chain]:
        return {
            int(pos_str): Chain.from_dict(chain_dict)  # type: ignore
            for pos_str, chain_dict in dic.items()
        }

    def get_data_type(self) -> type:
        return dict


class Pos2ChainMapNode(BaseNode):
    """
    Converts a list of NON-INTERSECTING chains into mapping (pos -> chain)
    """
    def __init__(self, name: str):
        super().__init__(name, [list], Pos2ChainMappingDescriptor())

    def process(self, chains: List[Chain]) -> Dict[int, Chain]:
        pos2chain: Dict[int, Chain] = {}
        for i, chain in enumerate(chains):
            for pos in chain.get_token_positions():
                pos2chain[pos] = chain

        return pos2chain
