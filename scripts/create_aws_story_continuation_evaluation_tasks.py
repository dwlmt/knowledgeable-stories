import argparse
import collections
import csv
import os
from random import random
from typing import List, OrderedDict

import fire
import more_itertools
import pandas
from jsonlines import jsonlines
from tqdm import tqdm
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter, SentenceSplitter


def cleanup_text(param):
    if param is None or len(param) == 0:
        return param
    for r in ["\n","<|endofsentence|>","<|endoftext|>","<newline>"]:
        param = param.replace(r, "")
    return param


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory is None or len(directory) == 0:
        return
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)

def create(prompts_json: str, gold_json: str, models_json: List[str], models_types: List[str],
           output_file: str, debug_prefix: bool = False,
           story_length=20,
           extra_columns=[],
           max_story_length=25):

    print("Input", models_json, models_types)

    ensure_dir(output_file)

    sentence_splitter: SentenceSplitter = SpacySentenceSplitter()

    if isinstance(models_json, str):
        models_json = [models_json]

    if isinstance(models_types, str):
        models_types = [models_types]

    # assert len(models_json) == len(models_types), "Models and types provided must be the same length."

    number_of_models = len(models_json) + 1

    prompt_dict = collections.OrderedDict()
    gold_dict = collections.OrderedDict()

    models_dict = collections.OrderedDict()

    length_list = []

    prompt_split = []
    with jsonlines.open(prompts_json) as reader:

        for obj in reader:
            clean_passage = cleanup_text(obj["passage"])
            prompt_split = sentence_splitter.split_sentences(clean_passage)

            prompt_dict[obj["story_id"]] = {"story_id": obj["story_id"], "passage": clean_passage, "sentences": prompt_split}

    with jsonlines.open(gold_json) as reader:

        for obj in reader:
            sentences = sentence_splitter.split_sentences(cleanup_text(obj["passage"]))

            if len(sentences) < max_story_length:
                continue

            prompt_sentences = sentences[:story_length]
            prompt_dict[obj["story_id"]]["passage"] = f"{prompt_dict[obj['story_id']]['passage']} {' '.join(prompt_sentences)}"
            prompt_dict[obj["story_id"]]["sentences"] += prompt_sentences

            continuation_sentences = sentences[story_length: max_story_length]
            sentence_text = " ".join(continuation_sentences)
            # sentence_text = f"<p>{sentence_text}</p>"

            # story_text = f"{prompt_text} {sentence_text}"
            story_text = f"{sentence_text}"

            length_list.append({"story_id": obj["story_id"], "type": "gold", "story_length_char": len(story_text)})
            gold_dict[obj["story_id"]] = {"story_id": obj["story_id"], "passage": story_text,
                                          "story_length_char": len(story_text)}

    models_dict["gold"] = gold_dict
    for m, t in zip(models_json, models_types):

        m_dict = collections.OrderedDict()
        with jsonlines.open(m) as reader:
            for obj in reader:
                if "story_id" in obj:
                    story_id = obj["story_id"]
                else:
                    story_id = obj["input"]["story_id"]

                m_dict[story_id] = {"story_id": story_id}

                sentences = []
                print(m,  obj["generated"])

                if len(obj["generated"]) == 0:
                    continue

                obj_sentences = obj["generated"][0]
                if "sentences" in obj_sentences:
                    obj_sentences = obj_sentences["sentences"]

                for sentence in obj_sentences:
                    print(m,t,sentence)
                    sentences.append(cleanup_text(sentence["text"]))

                sentences = sentences[story_length: ]
                sentence_text = " ".join(sentences)
                story_text = f"{sentence_text}"

                m_dict[story_id]["passage"] = story_text
                m_dict["story_length_char"] = len(story_text)

                length_list.append({"story_id": story_id, "type": t, "story_length_char": len(story_text)})


        models_dict[t] = m_dict

    print(f"Prompts: {prompt_dict.values()}")
    print(f"Gold: {gold_dict.values()}")
    print(f"Models: {models_dict.values()}")

    csv_rows = []

    for p_key, p_val in prompt_dict.items():
        csv_row_dict = collections.OrderedDict()
        csv_row_dict["story_id"] = p_key
        csv_row_dict["prompt"] = p_val["passage"]

        models_rows = []

        for model_key, model_val in models_dict.items():

            if p_key in model_val:

                if "passage" in model_val[p_key]:
                    passage = model_val[p_key]["passage"]

                    model_row_dict = {"type": model_key, "passage": passage}
                    models_rows.append(model_row_dict)

        if len(models_rows) == number_of_models:
            from random import shuffle
            shuffle(models_rows)

            for i, r in enumerate(models_rows, start=1):
                csv_row_dict[f"story_{i}"] = r["passage"]
                csv_row_dict[f"story_{i}_type"] = r["type"]

                if debug_prefix:
                    csv_row_dict[f"story_{i}"] = f"STORY TYPE DEBUG {r['type']} : " + csv_row_dict[f"story_{i}"]

            csv_rows.append(csv_row_dict)

    with open(output_file, 'w', newline='') as csv_file:

        csv_writer = csv.DictWriter(csv_file, fieldnames=list(csv_rows[0].keys()) + extra_columns, quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writeheader()

        for row in csv_rows:
            #print(f"Row: {row}")
            csv_writer.writerow(row)

    length_df = pandas.DataFrame(length_list)
    print(length_df)

    length_stats_df = length_df[["type","story_length_char"]].groupby("type").describe()
    print("Length Stats", length_stats_df)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(
    description='Run stats from  the prediction output, clustering and stats for the annotations and predictions.')
parser.add_argument('--prompts-json', required=True, type=str, help="The standalone prompts.")
parser.add_argument('--gold-json', required=True, type=str, help="The gold standard json.")
parser.add_argument('--output-file', required=True, type=str, help="The gold standard json.")
parser.add_argument('--story-length', required=False, type=int, default=20, help="Story length.")
parser.add_argument('--max-story-length', required=False, type=int, default=25, help="Max story length.")
parser.add_argument('--models-json', required=True, type=str, nargs="+", help="The models generated json output.")
parser.add_argument('--models-types', required=True, type=str, nargs="+", help="Types for the models.")
parser.add_argument('--extra-columns', required=False, type=str, nargs="+", default=[], help="Extra columns.")
parser.add_argument("--debug-prefix", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Add a debug prefix.")

args = parser.parse_args()

create(prompts_json=args.prompts_json, gold_json=args.gold_json, output_file=args.output_file,
       models_json=args.models_json, models_types=args.models_types, debug_prefix=args.debug_prefix,
       story_length=args.story_length, extra_columns=args.extra_columns,
       max_story_length=args.max_story_length)


