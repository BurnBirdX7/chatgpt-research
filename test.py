from typing import Dict

import torch
from transformers import RobertaTokenizer, RobertaModel
import wikipedia

"""
This code graps Elvis Presley page from Wikipedia and weeds in to RoBERTa to get token embeddings
Prints acquired data into embeddings.txt file
"""


def parse_wiki(title: str = "Elvis_Presley") -> str:
    wikipedia.set_lang("en")

    target_page = wikipedia.page(title)

    return target_page.content


def main():
    model_name = 'roberta-base'

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    # encoded_input = tokenizer(text, return_tensors='pt')
    # output = model(**encoded_input)

    tokens = tokenizer.tokenize(parse_wiki())
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    tensor_len: int = 512
    count: int = 1 + len(input_ids) // tensor_len
    print(f"Text is divided into {count} parts")

    with open('embeddings.txt', 'bw+') as f:
        for i in range(count):
            print(f"{i + 1} / {count}...", end='')
            tokens_slice = tokens[i:i + tensor_len]
            input_ids_slice = input_ids[i:i + tensor_len]

            input_ids_tensor = torch.tensor(input_ids_slice).unsqueeze(0)
            output = model(input_ids_tensor)

            embeddings = output.last_hidden_state.squeeze(0)

            for t, e in zip(tokens_slice, embeddings):
                f.write(f"{t}:\n".encode('utf-8'))
                f.write(f"{e!r}\n\n".encode('utf-8'))

            print("done")


if __name__ == '__main__':
    main()
