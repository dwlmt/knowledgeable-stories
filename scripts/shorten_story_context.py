import argparse
import os
from pathlib import Path

from allennlp.data.tokenizers import sentence_splitter
from jsonlines import jsonlines

def cleanup_text(param):
    if param is None or len(param) == 0:
        return param
    for r in ["\n","<|endofsentence|>","<|endoftext|>","<newline>"]:
        param = param.replace(r, "")
    return param

parser = argparse.ArgumentParser(
    description='Read the stories and write the first n sentences to a new file thereby truncating the story.')
parser.add_argument('--input-json', required=True, type=str, help="Read from this JSON file.")
parser.add_argument('--output-json', type=str, required=True, help="The JSON file to write to.")
parser.add_argument('--keep-sentences', type=int, required=False, default=20, help="Number of sentences to keep.")

args = parser.parse_args()


def process_text(args):

    Path(os.path.dirname(args["output_json"])).mkdir(parents=True, exist_ok=True)

    #print(f"Process json from ${args['input_json']}")

    json_list = []

    with jsonlines.open(args['input_json']) as reader:
        for obj in reader:

            clean_passage = cleanup_text(obj["passage"])
            prompt_split = sentence_splitter.split_sentences(clean_passage)

            prompt_split = prompt_split[:args["keep_sentences"]]
            obj["passage"] = " ".join(prompt_split)
            json_list.append(obj)

    with jsonlines.open(args['output_json'], mode='w') as writer:

        for j in json_list:
            writer.write(j)


process_text(vars(args))