from typing import Dict

import pandas as pd
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


"""
Prints acquired data into embeddings.txt file
"""


def main():
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)

    text = parse_wiki()

    # Should we add <s> </s> tags?
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    vector_len: int = 512
    padding_len: int = vector_len - (len(input_ids) % vector_len)
    input_ids += [tokenizer.pad_token_id] * padding_len  # add padding

    input_ids_tensor = torch.tensor(input_ids).reshape((-1, vector_len))
    output = model(input_ids_tensor)

    embeddings = output.last_hidden_state.detach()
    embeddings = embeddings.reshape((-1, embeddings.size()[2]))  # Squeeze batch dimension

    df = pd.DataFrame(embeddings)
    df.to_csv('embeddings.csv')


if __name__ == '__main__':
    main()
