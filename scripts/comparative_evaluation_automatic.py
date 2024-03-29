import argparse
import collections
import csv
import os
from typing import List

import more_itertools
import pandas
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter, SentenceSplitter
from datasets import load_metric
from jsonlines import jsonlines


def cleanup_text(param):
    if param is None or len(param) == 0:
        return param
    for r in ["\n", "<|endofsentence|>", "<|endoftext|>", "<newline>"]:
        param = param.replace(r, "")
    return param


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory is None or len(directory) == 0:
        return
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)


def eval(prompts_json: str, gold_json: str, models_json: List[str], models_types: List[str],
         output_dir: str, debug_prefix: bool = False,
         story_length=20, max_story_length=25):
    print("Input", models_json, models_types)

    ensure_dir(output_dir)

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

            prompt_dict[obj["story_id"]] = {"story_id": obj["story_id"], "passage": clean_passage,
                                            "sentences": prompt_split}

    with jsonlines.open(gold_json) as reader:

        for obj in reader:
            sentences = sentence_splitter.split_sentences(cleanup_text(obj["passage"]))
            # prompt_text =  " ".join(prompt_dict[obj["story_id"]]["sentences"])
            # prompt_text = f"<p><b>{prompt_text}</b></p>"

            # sentences = prompt_text + sentences
            if len(sentences) < max_story_length:
                continue

            prompt_sentences = sentences[:story_length]
            prompt_dict[obj["story_id"]]["passage"] += " ".join(prompt_sentences)
            prompt_dict[obj["story_id"]]["sentences"] += prompt_sentences

            sentences = sentences[story_length: max_story_length]
            story_text = " ".join(sentences)


            length_list.append({"story_id": obj["story_id"], "type": "gold", "story_length_char": len(story_text)})
            gold_dict[obj["story_id"]] = {"story_id": obj["story_id"], "passage": sentences,
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
                print(m, obj["generated"])

                if len(obj["generated"]) == 0:
                    continue

                obj_sentences = obj["generated"][0]
                if "sentences" in obj_sentences:
                    obj_sentences = obj_sentences["sentences"]

                for sentence in obj_sentences:
                    print(m, t, sentence)
                    sentences.append(cleanup_text(sentence["text"]))

                sentences = sentences[story_length: max_story_length]
                story_text = " ".join(sentences)

                m_dict[story_id]["passage"] = sentences
                m_dict["story_length_char"] = len(story_text)

                if len(story_text) > 0:
                    length_list.append({"story_id": story_id, "type": t, "story_length_char": len(story_text)})

        models_dict[t] = m_dict

    print(f"Prompts: {prompt_dict.values()}")
    print(f"Gold: {gold_dict.values()}")
    print(f"Models: {models_dict.values()}")

    aligned_rows = []

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

            for i, r in enumerate(models_rows, start=1):
                csv_row_dict[f"story_{r['type']}"] = r["passage"]

                if debug_prefix:
                    csv_row_dict[f"story_{r['type']}"] = f"STORY TYPE DEBUG {r['type']} : " + csv_row_dict[
                        f"story_{r['type']}"]

            aligned_rows.append(csv_row_dict)

    with open(f"{output_dir}/aws.csv", 'w', newline='') as csv_file:

        csv_writer = csv.DictWriter(csv_file, fieldnames=list(aligned_rows[0].keys()), quoting=csv.QUOTE_NONNUMERIC)
        csv_writer.writeheader()

        for row in aligned_rows:
            # print(f"Row: {row}")
            csv_writer.writerow(row)

    length_df = pandas.DataFrame(length_list)
    print(length_df)

    length_stats_df = length_df[["type", "story_length_char"]].groupby("type").describe()
    print("Length Stats", length_stats_df)

    # Do the pairwise comparison.
    pairwise_comparison_list = []
    model_permutations = more_itertools.distinct_permutations(models_dict.keys(), 2)

    for model_pair in model_permutations:

        meteor_total = 0.0
        bleu_total = 0.0
        bleurt_total = 0.0
        bertscore_total = 0.0

        model_pair_dict = collections.OrderedDict()
        model_pair_name = f"{model_pair[0]}_{model_pair[1]}"
        model_pair_dict["pair_name"] = model_pair_name
        model_pair_dict["model_one"] = model_pair[0]
        model_pair_dict["model_two"] = model_pair[1]

        sentences = list(range(1, max_story_length - story_length + 1, 1))
        num_extended = len(sentences)
        for i in sentences:

            # if model_pair[0] != "gold":
            #    continue

            meteor = load_metric("meteor")
            bleu = load_metric("sacrebleu")
            bleurt = load_metric('bleurt')
            # bertscore = load_metric('bertscore')

            model_1_texts = []
            model_2_texts = []

            for row in aligned_rows:

                print(row)
                if i > len(row[f"story_{model_pair[0]}"]) or  i > len(row[f"story_{model_pair[1]}"]):
                    continue

                model_1_text = row[f"story_{model_pair[0]}"][i-1]
                model_2_text = row[f"story_{model_pair[1]}"][i-1]

                model_1_texts.append(model_1_text)
                model_2_texts.append(model_2_text)

                meteor.add(prediction=model_2_text, reference=model_1_text)
                bleurt.add(prediction=model_2_text, reference=model_1_text)

            bleu.add_batch(predictions=model_2_texts, references=[[t] for t in model_1_texts])

            meteor_score = meteor.compute()
            model_pair_dict[f"meteor_score_{i}"] = meteor_score["meteor"]
            meteor_total += meteor_score["meteor"]

            bleu_score = bleu.compute()
            model_pair_dict[f"bleu_score_{i}"] = bleu_score["score"]
            bleu_total += bleu_score["score"]

            bleurt_score = bleurt.compute()
            print(bleurt_score)
            model_pair_dict[f"bleurt_score_{i}"] = sum(bleurt_score["scores"]) / len(bleurt_score["scores"])
            bleurt_total += sum(bleurt_score["scores"]) / len(bleurt_score["scores"])


            from bert_score import score
            P, R, F1 = score(model_2_texts, model_1_texts, lang='en', verbose=True)
            print("BERT", P, R, F1)
            model_pair_dict[f"bert_score_{i}"] = F1.mean().item()
            bertscore_total += F1.mean().item()

        model_pair_dict[f"bleu_score_all"] = bleu_total / num_extended
        model_pair_dict[f"meteor_score_all"] = meteor_total / num_extended
        model_pair_dict[f"bleurt_score_all"] = bleurt_total / num_extended
        model_pair_dict[f"bert_score_all"] = bertscore_total / num_extended

        print(model_pair_dict)

        pairwise_comparison_list.append(model_pair_dict)

        with open(f"{output_dir}/pairwise_metrics.csv", 'w', newline='') as csv_file:

            csv_writer = csv.DictWriter(csv_file, fieldnames=list(pairwise_comparison_list[0].keys()),
                                        quoting=csv.QUOTE_NONNUMERIC)
            csv_writer.writeheader()

            for row in pairwise_comparison_list:
                # print(f"Row: {row}")
                csv_writer.writerow(row)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


'''
parser = argparse.ArgumentParser(
    description='Run stats from  the prediction output, clustering and stats for the annotations and predictions.')
parser.add_argument('--prompts-json', required=True, type=str, help="The standalone prompts.")
parser.add_argument('--gold-json', required=True, type=str, help="The gold standard json.")
parser.add_argument('--output-dir', required=True, type=str, help="The gold standard json.")
parser.add_argument('--story-length', required=False, type=int, default=20, help="Story length.")
parser.add_argument('--max-story-length', required=False, type=int, default=25, help="Max story length.")
parser.add_argument('--models-json', required=True, type=str, nargs="+", help="The models generated json output.")
parser.add_argument('--models-types', required=True, type=str, nargs="+", help="Types for the models.")
parser.add_argument("--debug-prefix", type=str2bool, nargs='?',
                    const=True, default=False,
                    help="Add a debug prefix.")

args = parser.parse_args()

eval(prompts_json=args.prompts_json, gold_json=args.gold_json, output_dir=args.output_dir,
     models_json=args.models_json, models_types=args.models_types, debug_prefix=args.debug_prefix,
     story_length=args.story_length, max_story_length=args.max_story_length)

'''

eval(prompts_json="/home/s1569885/WP_prompts/test_prompts_1000.jsonl",
     gold_json="/home/s1569885/WP_prompts/test_gold_1000.jsonl",
     output_dir="./story_continuations_eval/",
     models_json=[
         "/home/s1569885/stories_continuation/story_context_wp_kb_big_text_medium_tdvae_big_wider_anneal_27306.jsonl",
         "/home/s1569885/stories_continuation/story_context_wp_kb_big_text_medium_tdvae_big_wider_memory_avg_30631.jsonl",
         "/home/s1569885/stories_continuation/story_context_wp_kb_big_text_medium_lstm.jsonl",
         "/home/s1569885/stories_continuation/story_context_wp_kb_big_text_medium_transformer_26599.jsonl",
         "/home/s1569885/stories_continuation/story_context_lm_big.jsonl"],
     models_types=["tdvae", "tdvae_memory", "lstm", "transformer", "lm"], debug_prefix=False,
     story_length=20, max_story_length=25)
