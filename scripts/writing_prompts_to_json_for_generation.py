''' A simple script for converting per line split text into passage prompts that can be used for story generation.

'''
import argparse

from jsonlines import jsonlines

parser = argparse.ArgumentParser(
    description='Extract text to the predictor format used in the new ')
parser.add_argument('--text-file', required=True, type=str, help="The SQLLite datase to read the stories from.")
parser.add_argument('--output-json', type=str, required=True, help="The location to save the output json files.")

args = parser.parse_args()


def process_text(args):

    print(f"Process text from ${args['text_file']}")

    json_list = []

    with open(args['output_json'], mode='w') as reader:
        for i, line in enumerate(reader):
            json_dict = {}
            json_dict["story_id"] = f"{i}"
            json_dict["passage"] = line

            json_list.append()


    with jsonlines.open(args['output_json'], mode='w') as writer:

        for j in json_list:
            writer.write(j)


process_text(vars(args))