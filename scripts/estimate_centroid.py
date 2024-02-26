from typing import List, Iterable

import numpy as np

from src import EmbeddingsBuilder, Wiki, Roberta
from transformers import RobertaTokenizer, RobertaModel     # type: ignore
from progress.bar import ChargingBar                        # type: ignore

from src.config.EmbeddingsConfig import EmbeddingsConfig
from src.config.WikiConfig import WikiConfig


def estimate_centroid(data: Iterable[str], tokenizer: RobertaTokenizer, model: RobertaModel) -> np.ndarray:
    embedding_config = EmbeddingsConfig(tokenizer=tokenizer,
                                        model=model,
                                        normalize=False,
                                        centroid_file=None)

    embedding_builder = EmbeddingsBuilder(embedding_config)
    embedding_builder.suppress_progress = True

    embeddings = np.empty((0, embedding_builder.embedding_length))
    for page in ChargingBar('Processing texts').iter(data):
        embeddings = np.concatenate([embeddings, embedding_builder.from_text(page)])

    return embeddings.mean(0)


# List of articles expected to be distanced from topic of Elvis Presley
article_list__unelvis = [
    "Stand-up_comedy",
    "FANUC",
    "Francis_Augustus_Cox",
    "(We_All_Are)_Looking_for_Home",
    "Computer_science",
    "3D_printing",
    "Thermoplastic"
] + [
    "Knut_Storberget",
    "One_of_Those_Nights_(Juicy_J_song)",
    "Politically_exposed_person",
    "Eulaema",
    "Struell_Wells",
    "Pollinator",
    "Sir_Alexander_Fleming_College",
    "Amy_Hughes_(artist)",
    "Jing_Lusi",
    "Recurring_Saturday_Night_Live_characters_and_sketches_introduced_2007-08",
    "Trout_Creek_Hill",
    "Shaynna_Blaze",
    "Leckhampstead,_Buckinghamshire",
    "Mu_Cassiopeiae",
    "Dave_Karnes",
    "Akron_Goodyear_Wingfoots",
    "Australian_cricket_team_in_India_in_2000-01",
    "Sergio_Hernandez_(basketball)",
    "Phil_Joanou",
    "Epiphany_Apostolic_College",
    "WGN-TV",
    "Jacob_Josefson",
    "We_Connect",
    "Tiare_Aguilera_Hey",
    "Apna_Bombay_Talkies",
    "Battle_of_Cravant",
    "So_This_Is_Paris_(1926_film)"
]


def estimate_centroid_for_elvis_persona(centroid_file: str, tokenizer: RobertaTokenizer, model: RobertaModel) -> None:
    page_names = WikiConfig.get_elvis_config().target_pages + article_list__unelvis
    texts: List[str] = []
    for name in ChargingBar('Loading articles').iter(page_names):
        texts += Wiki.parse(name).values()

    centroid_data = estimate_centroid(texts, tokenizer, model)
    np.save(centroid_file, centroid_data)

    print('Centroid:')
    print(repr(centroid_data))


if __name__ == '__main__':
    estimate_centroid_for_elvis_persona(*Roberta.get_default())
