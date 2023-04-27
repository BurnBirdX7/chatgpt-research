import faiss
import pandas as pd
import numpy as np
import torch
from IntervalToSource import IntervalToSource
from transformers import RobertaTokenizer, RobertaModel


def test_request(index, q):
    k = 4
    dist, ind = index.search(q, k)
    print(f"Distances to {k} nearest neighbours:")
    print(dist)
    print(f"Indexes of {k} nearest neighbours:")
    print(ind)


# from 'Childhood in Tupelo' section
childhood = "Presley's father Vernon was of German,[11] Scottish, and English origins,[12] and a descendant of the " \
            "Harrison family of Virginia through his mother, Minnie Mae Presley (née Hood).[8] Presley's mother " \
            "Gladys was Scots-Irish with some French Norman ancestry.[13] She and the rest of the family believed " \
            "that her great-great-grandmother, Morning Dove White, was Cherokee.[14][15][16] This belief was restated "\
            "by Elvis's granddaughter Riley Keough in 2017.[17] Elaine Dundy, in her biography, supports the belief.[" \
            "18]"
childhood_url = 'https://en.wikipedia.org/wiki/Elvis_Presley#Childhood_in_Tupelo'

legacy = "President Jimmy Carter remarked on Presley's legacy in 1977: \"His music and his personality, fusing the styles of white country and black rhythm and blues, permanently changed the face of American popular culture. His following was immense, and he was a symbol to people the world over of the vitality, rebelliousness, and good humor of his country.\""
legacy_url = 'https://en.wikipedia.org/wiki/Elvis_Presley#Legacy'


def test_wiki(index, text, expected_url):
    i2s = IntervalToSource.read_csv('ranges.csv')

    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    print('Getting embeddings for wiki text...', end='')
    ids = tokenizer.encode(text)
    input_ids = torch.tensor(ids).unsqueeze(0)
    output = model(input_ids)
    embeddings = output.last_hidden_state.detach().squeeze(0).numpy()
    print(' Done')
    print(embeddings.shape)

    k = 1
    result_dists, result_idxs = index.search(embeddings, k)
    expected_count = 0
    for i, (token_dists, token_idxs) in enumerate(zip(result_dists, result_idxs)):
        print(f'Token #{i+1} info...:')
        for dist, idx in zip (token_dists, token_idxs):
            print(f'Distance: {dist}, ', end='')
            print(f'ID: {idx}, ', end='')
            src = i2s.get_source(idx)
            print(f'Source: {src}')

            if src == expected_url:
                expected_count += 1
        print('---')

    print(f'Expectation met: {expected_count / len(result_dists) / k * 100}%')


def main():
    print("Loading embeddings...", end='')
    data = np.array(pd.read_csv('embeddings.csv'),
                    order='C', dtype=np.float32)  # C-contiguous order and np.float32 type are required
    sequence_len, embedding_len = data.shape
    print('Done')
    print(data.shape)

    print("Building index...", end="")
    index = faiss.IndexFlatL2(embedding_len)
    index.add(data)
    print("Done")

    print("Searching first 5 embeddings...")
    test_request(index, data[:5])

    print("Searching last 5 embeddings...")
    test_request(index, data[-5:])

    test_wiki(index, childhood, childhood_url)
    test_wiki(index, legacy, legacy_url)


if __name__ == '__main__':
    main()
