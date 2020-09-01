import argparse
import collections
import csv
import json
import os
from random import random
from typing import List, OrderedDict

import fire
import more_itertools
import pandas
from jsonlines import jsonlines
from tqdm import tqdm
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter, SentenceSplitter

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if directory is None or len(directory) == 0:
        return
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)

ASSIGNMENT_COL = 'AssignmentId'
WORKER_COL = 'WorkerId'
HIT_COL = 'HITId'
ANSWER_COL = 'Answer.taskAnswers'

def evaluate(aws_results, output_dir, number_of_story_types):

    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None)

    print(f"Evaluate AWS results from: {aws_results}")

    ensure_dir(output_dir)
    print(f"Save AWS evaluation to: {output_dir}")

    df = load_dfs(aws_results)

    print(df)
    print("Columns", df.columns)

    deanonymised_list = []

    for index, row in df.iterrows():

        for i in range(1, number_of_story_types + 1):

            d_dict = {}

            d_dict["story_type"] = row[f"Input.story_{i}_type"]
            d_dict["story_text"] = row[f"Input.story_{i}"]
            d_dict["prompt"] = row['Input.prompt']
            d_dict[WORKER_COL] = row[WORKER_COL]
            d_dict[ASSIGNMENT_COL] = row[ASSIGNMENT_COL]
            d_dict[HIT_COL] = row[HIT_COL]

            json_answers = json.loads(row[ANSWER_COL])[0]
            print(json_answers)

            d_dict["rationale"] = json_answers["rationale"]

            # Swap around the ranking to story ranking.
            for r in range(1, number_of_story_types + 1):

                for t in ["overall","coherence","relevance","style","suspense"]:
                    value = json_answers[f"{t}_ranking_{r}"]
                    if value == i:
                        print(f"{t}_ranking",value)
                        d_dict[f"{t}_ranking"] = value

            deanonymised_list.append(d_dict)


    print(deanonymised_list)
    story_df = pandas.DataFrame(deanonymised_list)
    print(story_df)


def load_dfs(aws_results):
    df_list = []
    for res in aws_results:
        df = pandas.read_csv(res)
        df_list.append(df)
    df = pandas.concat(df_list)
    return df


parser = argparse.ArgumentParser(
    description='Post process and produce the evaluation metrics from AWS.')

parser.add_argument('--aws-results', required=True, type=str, nargs="+", help="List of AWS results.")
parser.add_argument('--output-dir', required=True, type=str, help="The output dir for the results.")
parser.add_argument('--number-of-story-types', required=False, type=int, default=5, help="Number of stories in the AWS evaluation.")

args = parser.parse_args()

evaluate(aws_results=args.aws_results, output_dir=args.output_dir, number_of_story_types=args.number_of_story_types)


