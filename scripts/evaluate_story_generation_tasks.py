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

def evaluate(aws_results, output_dir):

    pandas.set_option('display.max_rows', None)
    pandas.set_option('display.max_columns', None)

    print(f"Evaluate AWS results from: {aws_results}")

    ensure_dir(output_dir)
    print(f"Save AWS evaluation to: {output_dir}")

    df = load_dfs(aws_results)

    print(df)
    print("Columns", df.columns)


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

args = parser.parse_args()

evaluate(aws_results=args.aws_results, output_dir=args.output_dir)


