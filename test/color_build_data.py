import faiss
import collections

from scripts.model_of_GPT import build_page_template

from src.embeddings import text_embedding
from src import SourceMapping, Roberta, Config
from typing import Dict, List, Tuple
from src import SourceMapping, Roberta, Config, Embeddings, Index
from typing import Dict, List, Optional

tokenizer, model = Roberta.get_default()

# from 'Childhood in Tupelo' section
childhood_w_refs = (
    "Presley's father Vernon was of German, Scottish, and English origins,[12] and a descendant of the "
    "Harrison family of Virginia through his mother, Minnie Mae Presley (née Hood).[8] Presley's mother "
    "Gladys was Scots-Irish with some French Norman ancestry.[13] She and the rest of the family believed "
    "that her great-great-grandmother, Morning Dove White, was Cherokee.[14][15][16] This belief was restated "
    "by Elvis's granddaughter Riley Keough in 2017.[17] Elaine Dundy, in her biography, supports the belief.["
    "18]"
)

childhood_url = "https://en.wikipedia.org/wiki/Elvis_Presley#Childhood_in_Tupelo"


def build_dict_for_color(links: list[str], uniq_color: int) -> Dict[str, str]:
    filtered_links = [link for link in links if link is not None]
    dictionary_of_links = dict(collections.Counter(filtered_links))
    sorted_dict = dict(sorted(dictionary_of_links.items(), key=lambda x: x[1], reverse=True))
    links_with_uniq_colors = dict(list(sorted_dict.items())[:uniq_color])
    uniq_color_dict = {
        'Fuchsia': 'color1',
        'MediumPurple': 'color2',
        'DarkViolet': 'color3',
        'DarkMagenta': 'color4',
        'Indigo': 'color5'
    }

    for link, (_, color_hex) in zip(links_with_uniq_colors, uniq_color_dict.items()):
        links_with_uniq_colors[link] = color_hex

    return links_with_uniq_colors


def prob_test_wiki_with_colored(index: faiss.Index,
                                src_map: SourceMapping,
                                text: str,
                                expected_url: str,
                                uniq_color: int) -> None:
    embeddings = Embeddings(tokenizer, model).from_text(text)
def prob_test_wiki_with_colored(index: faiss.Index, src_map: SourceMapping, text: str, expected_url: str,
                                uniq_color: int) -> tuple[str, str, str]:
    embeddings = text_embedding(text, tokenizer, model)
    faiss.normalize_L2(embeddings)

    result_dists, result_ids = index.search(embeddings, 1)
    expected_count: int = 0
    dist_sum: float = 0.0

    intervalToSource = SourceMapping()
    ranges = intervalToSource.read_csv(Config.mapping_file)
    links: List[Optional[str]] = []

    for i, (token_dists, token_ids) in enumerate(zip(result_dists, result_ids)):

        dist = token_dists[0]
        idx = token_ids[0]

        if dist < Config.threshold:
            links.append(None)
        else:
            link = ranges.get_source(index=token_ids[0])
            links.append(link)

        src = src_map.get_source(idx)

        if src == expected_url:
            expected_count += 1
            dist_sum += dist

    print(
        f"Got expected URL in {expected_count / len(result_dists) * 100:.4f}% of cases, "
        f"average match distance: {dist_sum / len(result_dists):.4f}"
    )

    dict_with_uniq_colors = build_dict_for_color(links, uniq_color)

    return build_page_template(text, links, dict_with_uniq_colors)


def main(user_input: str) -> tuple[str, str, str]:
    sanity_test: bool = True
    read_index: bool = False

    if read_index:
        print("Readings index... ", end='')
        index = Index.load(Config.index_file, Config.mapping_file)
        print("Done")
    else:
        print("Index is being built from wiki... ")
        index = Index.from_wiki()

    print("Test [Data] Searching quotes from the same page:")
    print('"Childhood w references"')
    return prob_test_wiki_with_colored(index, index.mapping, user_input, childhood_url, 5)


if __name__ == "__main__":
    main()
