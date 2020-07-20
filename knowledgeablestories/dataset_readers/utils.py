import re
import string
from itertools import groupby

from allennlp.data.fields import TextField, ListField, LabelField
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

printable = set(string.printable)
allowed_punct_set = {".",",","'",'""',":",";","-","!","?","[","]","{","}","(",")","*","/","\\","_", "‘","’","“","”"}
def cleanup_text(text, ascii=True):
    
    mod_text = text.replace("\n", " ")
    if ascii:
        mod_text = re.sub(r'[^\x00-\x7F]', ' ', mod_text)
    else:
        mod_text = ''.join(filter(lambda x: x in printable, mod_text))

    mod_text = "".join([c for c in mod_text if c.isalnum() or c.isspace() or c in allowed_punct_set])

    return " ".join([t for t in mod_text.split() if len(t) < 25])

def position_to_labels_field(scalar_values):

    labels_list = []
    cat_ints = [min(max(int(s / 0.05) - 1, 0),19) for  s in scalar_values]
    for cat in cat_ints:
        labels_list.append(LabelField(cat,skip_indexing=True))
    field = ListField(labels_list)
    return field

def type_to_labels_field(scalar_value, number_of_sentences):

    labels_list = []
    for i in range(number_of_sentences):
        labels_list.append(LabelField(scalar_value,skip_indexing=True))
    field = ListField(labels_list)
    return field

def sentiment_to_labels_field(scalar_values):

    def map_sentiment(scalar):
        band_increment = float(2.0 / 9.0)
        if -1.0 + (band_increment * 1) <= scalar:
            return 0
        elif -1.0 + (band_increment * 2) <= scalar:
            return 1
        elif -1.0 + (band_increment * 3) <= scalar:
            return 2
        elif -1.0 + (band_increment * 4) <= scalar:
            return 3
        elif -1.0 + (band_increment * 5) <= scalar:
            return 4
        elif -1.0 + (band_increment * 6) <= scalar:
            return 5
        elif -1.0 + (band_increment * 7) <= scalar:
            return 6
        elif -1.0 + (band_increment * 8) <= scalar:
            return 7
        else:
            return 8

    labels_list = []
    cat_ints = [map_sentiment(s) for  s in scalar_values]
    for cat in cat_ints:
        labels_list.append(LabelField(cat,skip_indexing=True))
    field = ListField(labels_list)
    return field



def convert_to_textfield(text_batch, tokenizer, max_token_len, token_indexers):
    text_field_list = []
    for text in text_batch:
        tokens = tokenizer.tokenize(f"{text} <|endoftext|>")
        if len(tokens) > max_token_len:
            tokens = tokens[0: max_token_len]
        text_field_list.append(
            TextField(tokens, token_indexers=token_indexers))
    text_list_field = ListField(text_field_list)
    return text_list_field



def group_into_n_sentences(text, n):
    return [" ".join(text[i * n:(i + 1) * n]) for i in range((len(text) + n - 1) // n)]

punc = set(string.punctuation) - set('.')

def strip_repeating_punctuation(tokens):
    # Strip repeating characters.
    newtext = []
    for k, g in groupby(tokens):
        if k in punc:
            newtext.append(k)
        else:
            newtext.extend(g)
    tokens = ''.join(newtext)
    return tokens