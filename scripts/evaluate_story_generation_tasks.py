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
import pingouin as pg
from jsonlines import jsonlines
from scipy.stats import kendalltau, spearmanr, pearsonr
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

COPY_COLUMNS = ['HITTypeId', 'Title', 'Description', 'Keywords', 'Reward',
       'CreationTime', 'MaxAssignments', 'RequesterAnnotation',
       'AssignmentDurationInSeconds', 'AutoApprovalDelayInSeconds',
       'Expiration', 'NumberOfSimilarHITs', 'LifetimeInSeconds',
       'AssignmentId', 'WorkerId', 'AssignmentStatus', 'AcceptTime',
       'SubmitTime', 'AutoApprovalTime', 'ApprovalTime', 'RejectionTime',
       'RequesterFeedback', 'WorkTimeInSeconds', 'LifetimeApprovalRate',
       'Last30DaysApprovalRate', 'Last7DaysApprovalRate', 'Approve', 'Reject']

def evaluate(aws_results, output_dir, number_of_story_types, questions):

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

            d_dict["model_type"] = row[f"Input.story_{i}_type"]
            d_dict["story_text"] = row[f"Input.story_{i}"]
            d_dict["prompt"] = row['Input.prompt']
            d_dict[WORKER_COL] = row[WORKER_COL]
            d_dict[ASSIGNMENT_COL] = row[ASSIGNMENT_COL]
            d_dict[HIT_COL] = row[HIT_COL]

            for c in COPY_COLUMNS:
                d_dict[c] = row[c]

            json_answers = json.loads(row[ANSWER_COL])[0]
            print(json_answers)

            d_dict["rationale"] = json_answers["rationale"]

            # Swap around the ranking to story ranking.
            for r in range(1, number_of_story_types + 1):

                for t in ["overall","coherence","relevance","style","suspense"]:
                    value = json_answers[f"{t}_ranking_{r}"]
                    if int(value) == i:
                        print(f"{t}_ranking",r)
                        d_dict[f"{t}_ranking"] = int(r)

            deanonymised_list.append(d_dict)


    print(deanonymised_list)
    story_df = pandas.DataFrame(deanonymised_list)
    story_df.to_csv(f"{output_dir}/processed.csv")

    summary_stats(output_dir, story_df)

    anova_and_tukey(output_dir, story_df, questions)

    question_permutations = more_itertools.distinct_permutations(questions, r=2)
    rank_correlation_list = []
    for p in question_permutations:
        rank_dict = {}
        print(p[0],p[1])
        rank_dict["model_type_1"] = p[0]
        rank_dict["model_type_2"] = p[1]

        values_one = story_df[f"{p[0]}_ranking"]
        values_two = story_df[f"{p[1]}_ranking"]

        kendall, kendall_p_value = kendalltau(values_one, values_two)
        rank_dict["kendall"] = kendall
        rank_dict["kendall_p_value"] = kendall_p_value

        spearman, spearman_p_value = spearmanr(values_one, values_two)
        rank_dict["spearman"] = spearman
        rank_dict["spearman_p_value"] = spearman_p_value

        pearson, pearson_p_value = pearsonr(values_one, values_two)
        rank_dict["pearson"] = pearson
        rank_dict["pearson_p_value"] = pearson_p_value

        rank_correlation_list.append(rank_dict)

    rank_correlation_df = pandas.DataFrame(rank_correlation_list)
    print("Rank Correlation", rank_correlation_df)
    rank_correlation_df.to_csv(f"{output_dir}/questions_rank_correlation.csv")
            

def anova_and_tukey(output_dir, story_df, questions):
    for t in questions:
        value_col = f"{t}_ranking"

        aov = pg.anova(dv=value_col, between='model_type', data=story_df,
                       detailed=True).round(4)

        print("ANOVA", aov)
        aov.to_csv(f"{output_dir}/{value_col}_anova.csv")

        tukey = pg.pairwise_tukey(dv=value_col, between='model_type', data=story_df).round(4)
        print("TUKEY", tukey)
        tukey.to_csv(f"{output_dir}/{value_col}_tukey.csv")


def summary_stats(output_dir, story_df):
    story_type_summary_statistics = story_df.groupby('model_type').describe().unstack(1)
    print(story_type_summary_statistics)
    story_type_summary_statistics.to_csv(f"{output_dir}/story_type_summary_stats.csv")
    worker_summary_statistics = story_df.groupby(WORKER_COL).describe().unstack(1)
    print(worker_summary_statistics)
    worker_summary_statistics.to_csv(f"{output_dir}/worker_summary_stats.csv")
    hit_summary_statistics = story_df.groupby(HIT_COL).describe().unstack(1)
    print(hit_summary_statistics)
    story_type_summary_statistics.to_csv(f"{output_dir}/hit_summary_stats.csv")


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
parser.add_argument('--questions', required=False, type=str, default=["overall", "coherence", "relevance", "style", "suspense"])


args = parser.parse_args()

evaluate(aws_results=args.aws_results, output_dir=args.output_dir, number_of_story_types=args.number_of_story_types,
         questions=args.questions)


