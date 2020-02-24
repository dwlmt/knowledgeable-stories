from allennlp.data.fields import TextField, ListField
from whatthelang import WhatTheLang

wtl = WhatTheLang()

def is_english(text: str):
    try:
        lang = wtl.predict_lang(text)
        if not isinstance(lang, str):
            lang = "UKN"
    except:
        lang = "UKN"

    if lang == "en":
        return True
    else:
        return False


def removeNonAscii(s): return "".join(i for i in s if ord(i)<128)

def convert_to_textfield(tokens, tokenizer, max_token_len, token_indexers):
    text_field_list = []
    for tokens in tokens:
        tokens = tokenizer.tokenize(tokens)
        if len(tokens) > max_token_len:
            tokens = tokens[0: max_token_len]
        text_field_list.append(
            TextField(tokens, token_indexers=token_indexers))
    text_list_field = ListField(text_field_list)
    return text_list_field

def group_into_n_sentences(text, n):
    return [" ".join(text[i * n:(i + 1) * n]) for i in range((len(text) + n - 1) // n)]