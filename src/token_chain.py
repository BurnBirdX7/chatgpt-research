from __future__ import annotations

import copy
import math
import torch
from typing import Optional, List, Set, Any, Dict

from src.config import EmbeddingBuilderConfig
from src.pipeline import BaseNode, BaseDataDescriptor, base_data_descriptor


class Chain:
    begin_pos: Optional[int]
    end_pos: Optional[int]
    likelihoods: List[float]
    skips: int = 0
    source: str

    def __init__(self, source: str,
                 begin_pos: Optional[int] = None,
                 end_pos: Optional[int] = None,
                 likelihoods: Optional[List[float]] = None,
                 skips: int = 0):
        self.begin_pos = begin_pos
        self.end_pos = end_pos
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
        score = 1.0
        for lh in self.likelihoods:
            score *= lh

        score **= 1 / len(self.likelihoods)
        score *= math.log2(2 + len(self))
        return score

    @staticmethod
    def generate_chains(source_len: int, likelihoods: torch.Tensor,
                        token_ids: List[int], token_start_pos: int, source: str) -> List[Chain]:
        """
        Generates chains of tokens with the same source
        """
        result_chains: List[Chain] = []

        for source_start_pos in range(0, source_len):
            chain = Chain(source)
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
            Chain.from_dict(d)
            for d in dic["chains"]
        ]

    def get_data_type(self) -> type[list]:
        return list


class ChainingNode(BaseNode):
    def __init__(self, name: str, embedding_builder_config: EmbeddingBuilderConfig):
        super().__init__(name, [str, list, dict], ChainListDescriptor())
        self.en_config = embedding_builder_config

    def process(self, input_text: str, sources: List[str], sources_data: Dict[str, str], *ignore) -> Any:
        input_tokens = self.en_config.tokenizer.tokenize(input_text)
        input_token_ids = self.en_config.tokenizer.convert_tokens_to_ids(input_tokens)

        result_chains = []
        for token_pos, (token, token_id, source) in enumerate(zip(input_tokens, input_token_ids, sources)):
            wiki_text = sources_data[source]
            wiki_token_ids = self.en_config.tokenizer.encode(wiki_text, return_tensors='pt').squeeze()
            print(f"> token: '{token}', "
                  f"id: {token_id}, "
                  f"source token count: {len(wiki_token_ids)}, "
                  f"top source: {source}")

            for batch in range(0, len(wiki_token_ids), 256):
                print(f"\tbatch: [{batch} : {batch + 512})")
                wiki_token_ids_batch = wiki_token_ids[batch:batch + 512]
                if len(wiki_token_ids_batch) < 2:
                    break

                wiki_token_ids_batch = wiki_token_ids_batch.unsqueeze(0)

                with torch.no_grad():
                    output_page = self.en_config.model(wiki_token_ids_batch)

                wiki_logits = output_page[0].squeeze()
                likelihoods = torch.nn.functional.softmax(wiki_logits, dim=1)
                result_chains += Chain.generate_chains(len(wiki_logits), likelihoods, input_token_ids, token_pos,
                                                       source)

        return result_chains
