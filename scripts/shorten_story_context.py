import argparse
import os
from pathlib import Path

from jsonlines import jsonlines

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
            print(obj)

    """
    with jsonlines.open(args['output_json'], mode='w') as writer:

        for j in json_list:
            writer.write(j)
    """


process_text(vars(args))