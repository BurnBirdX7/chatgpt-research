

import wikipedia

wikipedia.set_lang("en")

target_page = wikipedia.page("Elvis_Presley")

with open('text_from_wiki.txt', 'w', encoding='utf-8') as f:
    f.write(target_page.content)
