import math
import time
import numpy as np  # type: ignore
import faiss  # type: ignore
import wikipediaapi  # type: ignore
import torch

from src import Roberta, Config, SourceMapping, Embeddings, Index, Wiki
from transformers import RobertaTokenizer, RobertaForMaskedLM

tokenizer, model = Roberta.get_default()
modelMLM = RobertaForMaskedLM.from_pretrained('roberta-large')
batched_token_ids = torch.empty((1, 512), dtype=torch.int)

mask_token = 50264

result_sequence = []


def get_page_section_from_wiki(source: str) -> str:
    wikipedia = wikipediaapi.Wikipedia("en")

    title = source.split("https://en.wikipedia.org/wiki/")[1].split("#")[0].replace("_", " ")

    target_page = wikipedia.page(title)
    section = target_page.section_by_title \
        (source.split("https://en.wikipedia.org/wiki/")[1].split("#")[1].replace("_",
                                                                                 " "))  # возврощает последнюю секцию

    return str(section)


def make_chain_colored_recursive_right(new_chain, last_hidden_state, hidden_state_start, tokens, token_pos, result_ids,
                                       chain_count):
    for hidden_state in range(hidden_state_start, 512):
        continue_chain = new_chain.copy()
        probs = torch.nn.functional.softmax(last_hidden_state[hidden_state])

        column_tokens = torch.topk(last_hidden_state[hidden_state], k=20, dim=0)
        top_twenty_tokens = column_tokens[1]
        probability_top_twenty_tokens = [probs[i] for i in top_twenty_tokens]
        words = [tokenizer.decode(i.item()).strip() for i in top_twenty_tokens]

        token = tokens[token_pos]
        if "Ġ" in token:
            token = token.split("Ġ")[1]

        for word in range(len(words)):
            if words[word] == token and probability_top_twenty_tokens[word].item() >= 0.1:
                chain_count+=1
                # log2(2+len)((lik_h_0*...*lik_h_len)^1/len) - score
                score = math.log2(chain_count)*math.pow(continue_chain[1]*probability_top_twenty_tokens[word].item(),
                                                        1/chain_count)
                continue_chain[0]=continue_chain[0]+token
                continue_chain[1]=score
                # continue_chain[2]=str(continue_chain[2])+"|"+str(result_ids[token_pos].item())
                tokens_in_chain = continue_chain[2].copy()
                tokens_in_chain.append(token_pos)
                continue_chain[2]=tokens_in_chain

                # continue_chain.append(token)
                # continue_chain.append(probability_top_twenty_tokens[word].item())
                # continue_chain.append(result_ids[token_pos].item())

                result_sequence.append(continue_chain)
                if hidden_state != len(last_hidden_state) - 1 and token_pos != len(tokens) - 1:
                    make_chain_colored_recursive_right(continue_chain, last_hidden_state, hidden_state + 1, tokens,
                                                   token_pos + 1, result_ids, chain_count)
            else:
                return


def iterate_on_sorted_sequence(iterator_sorting, filtered_elements):
    filtered_elements_new = []
    for i in range(iterator_sorting):
        filtered_elements_new.append(filtered_elements[i])

    first_elements = filtered_elements[iterator_sorting]
    for i, chain in enumerate(filtered_elements):
        if i > iterator_sorting:
            flag = False
            for j, seq_item in enumerate(chain[2]):
                for k, first_item in enumerate(first_elements[2]):
                    if seq_item == first_item:
                        flag = True
            if flag is False:
                filtered_elements_new.append(filtered_elements[i])


    print("res_first", filtered_elements_new)
    if iterator_sorting!=len(filtered_elements_new):
        iterator_sorting+=1
        iterate_on_sorted_sequence(iterator_sorting, filtered_elements_new)

    return

def main() -> None:
    index = Index.load(Config.index_file, Config.mapping_file)

    gpt_response = " Elvis Presley's father"  # gpt output

    embeddings = Embeddings(tokenizer, model).from_text(gpt_response)
    print(embeddings)
    faiss.normalize_L2(embeddings)

    result_dists, result_ids = index.index.search(embeddings, 1)
    print("indexes:", result_ids, "result_ids dists:", result_dists, "\n\n")

    tokens = tokenizer.tokenize(gpt_response)  # разбиваем на токены входную строку с гпт
    print("tokens:", tokens, "\n\n")  # все токены разбитые из input

    start = time.perf_counter()
    for token_pos, token in enumerate(tokens):
        source = index.get_source(int(result_ids[token_pos]))
        print("source: ", source)
        text_from_section = get_page_section_from_wiki(source)  # последняя секция из вики с документа

        if "Ġ" in token:
            token = token.split("Ġ")[1]
        print("token:", token)

        token_ids = tokenizer.encode(text_from_section, return_tensors='pt')

        for batch in range(0, token_ids.shape[1], 511):
            batched_token_ids = token_ids[0, batch:batch + 512].unsqueeze(0)
            with torch.no_grad():
                output = modelMLM(batched_token_ids)

            last_hidden_state = output[0].squeeze()
            print("res:::", last_hidden_state)
            print("shape:::", len(last_hidden_state))

            for hidden_state in range(len(last_hidden_state)):
                local_sequence = []

                probs = torch.nn.functional.softmax(last_hidden_state[hidden_state])
                column_tokens = torch.topk(last_hidden_state[hidden_state], k=20, dim=0)
                top_twenty_tokens = column_tokens[1]
                probability_top_twenty_tokens = [probs[i] for i in top_twenty_tokens]
                words = [tokenizer.decode(i.item()).strip() for i in top_twenty_tokens]

                for word in range(len(words)):
                    token_pos_in_chain = []
                    if words[word] == token and probability_top_twenty_tokens[word].item() >= 0.1:
                        chain_count = 1
                        local_sequence.append(token)
                        local_sequence.append(probability_top_twenty_tokens[word].item())  # first score
                        # local_sequence.append(result_ids[token_pos].item())

                        token_pos_in_chain.append(token_pos)
                        local_sequence.append(token_pos_in_chain)
                        local_sequence.append(source)

                        result_sequence.append(local_sequence)

                        if hidden_state != len(last_hidden_state) - 1 and token_pos != len(tokens) - 1:
                            make_chain_colored_recursive_right(local_sequence, last_hidden_state, hidden_state + 1,
                                                               tokens, token_pos + 1, result_ids, chain_count)

            print("probe res::", result_sequence)

    print("whole res::", result_sequence)
    print("shape_of_end_sequence:::", len(result_sequence))

    sorted_result_sequence = sorted(result_sequence, key=lambda x: x[1], reverse=True)
    print("sorting:::", sorted_result_sequence)

    filtered_elements = []
    iterator_sorting = 1
    first_elements = sorted_result_sequence[0]

    filtered_elements.append(first_elements)
    for i, chain in enumerate(sorted_result_sequence):
        if i != 0:
            flag=False
            for j, seq_item in enumerate(chain[2]):
                for k, first_item in enumerate(first_elements[2]):
                    if seq_item == first_item:
                        flag=True
            if flag is False:
                filtered_elements.append(sorted_result_sequence[i])

    print("res_first", filtered_elements)
    if iterator_sorting != len(filtered_elements):
        iterator_sorting+=1
        iterate_on_sorted_sequence(iterator_sorting, filtered_elements)

    print("whole time:", time.perf_counter() - start)


if __name__ == "__main__":
    main()
