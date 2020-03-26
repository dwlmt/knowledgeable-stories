import argparse
import collections
import json
import os
from collections import OrderedDict, Iterable
from io import StringIO
from itertools import combinations

import numpy
import pandas
import pandas as pd
import plotly
import plotly.graph_objs as go
import plotly.io as pio
import scipy
import torch
from jsonlines import jsonlines
from nltk import interval_distance, AnnotationTask
from scipy.spatial import distance
from scipy.stats import kendalltau, pearsonr, spearmanr, combine_pvalues
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Run stats from  the prediction output, clustering and stats for the annotations and predictions.')
parser.add_argument('--prediction-json', required=False, type=str, help="The JSON files with the predictions.")
parser.add_argument('--annotator-targets', required=False, type=str, help="CSV with consensus predictions.")
parser.add_argument('--output-dir', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument("--no-html-plots", default=False, action="store_true", help="Don't save plots to HTML")
parser.add_argument("--no-pdf-plots", default=False, action="store_true", help="Don't save plots to PDF")
parser.add_argument("--folds", default=5, type=int, help="Folds in the cross validation.")
parser.add_argument("--epochs", default=25, type=int, help="Number of Epochs for model fitting.")
parser.add_argument("--exclude-worker-ids", type=str, nargs="*", help="Workers to exclude form the annotations.")
parser.add_argument("--cuda-device", default=0, type=int, help="The default CUDA device.")
parser.add_argument("--export-only", default=False, action="store_true",
                    help="Only for export so remove legend and filter down export parameters.")
parser.add_argument('--export-columns', required=False, type=str, nargs="*",
                    default=[])

args = parser.parse_args()

metric_columns = [
    "corpus_surprise_gpt_embedding",
]

annotator_prediction_column = "suspense"


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def array_to_first_value(target_value):
    changed_value = target_value
    if isinstance(target_value, (numpy.ndarray, numpy.generic, torch.Tensor, pandas.Series)):
        changed_value = target_value.tolist()

    if isinstance(changed_value, list):
        changed_value = changed_value[0]

    return changed_value


def agreement(agreement_triples, m, results_dict, distance=interval_distance):
    if len(agreement_triples) > 2 and len(set([t[0] for t in agreement_triples])) > 1:
        t = AnnotationTask(data=agreement_triples, distance=distance)
        results_dict[f"{m}_alpha"] = t.alpha()
        results_dict[f"{m}_agreement"] = t.avg_Ao()


def features(feature_col, train_df):
    train_features = train_df[f"{feature_col}_scaled"].to_numpy()
    train_features = train_features.reshape(-1, 1)
    return train_features


class RelativeToAbsoluteModel(torch.nn.Module):

    def __init__(self, origin_weight=0.0, big_decrease=-0.30, decrease=-0.10, same=0.0, increase=0.10,
                 big_increase=0.30,
                 epsilon=0.05):
        super(RelativeToAbsoluteModel, self).__init__()
        self.origin = torch.nn.Linear(1, 1, bias=False)
        self.big_decrease = torch.nn.Linear(1, 1, bias=False)
        self.decrease = torch.nn.Linear(1, 1, bias=False)
        self.same = torch.nn.Linear(1, 1, bias=False)
        self.increase = torch.nn.Linear(1, 1, bias=False)
        self.big_increase = torch.nn.Linear(1, 1, bias=False)

        # Set the default weights. Grads can be turned off to run these
        torch.nn.init.constant_(self.origin.weight, origin_weight)
        torch.nn.init.constant_(self.big_decrease.weight, big_decrease)
        torch.nn.init.constant_(self.same.weight, same)
        torch.nn.init.constant_(self.decrease.weight, decrease)
        torch.nn.init.constant_(self.increase.weight, increase)
        torch.nn.init.constant_(self.big_increase.weight, big_increase)

        self.origin.requires_grad = False
        self.same.requires_grad = False

        self.epsilon = epsilon

    def forward(self, annotation_series):
        initial = self.origin(torch.tensor([1.0]))

        res_tensor = torch.tensor(initial)

        for cat in annotation_series:

            last = res_tensor[-1]

            cat = cat.item()

            if cat == 1:
                change_value = self.big_decrease(torch.tensor([1.0])).clamp(
                    max=min(0.0, self.decrease(torch.tensor([1.0])).item())) - self.epsilon
            elif cat == 2:
                change_value = self.decrease(torch.tensor([1.0])).clamp(
                    min=self.big_decrease(torch.tensor([1.0])).item() + self.epsilon, max=0.0 - self.epsilon)
            elif cat == 3:
                change_value = self.same(torch.tensor([1.0]))
            elif cat == 4:
                change_value = self.increase(torch.tensor([1.0])).clamp(min=0.0 + self.epsilon, max=self.big_increase(
                    torch.tensor([1.0])).item() - self.epsilon)
            elif cat == 5:
                change_value = self.big_increase(torch.tensor([1.0])).clamp(
                    min=max(0.0, self.increase(torch.tensor([1.0])).item()) + self.epsilon)

            if cat != 0:
                new_value = last + change_value
                res_tensor = torch.cat((res_tensor, new_value))

        return res_tensor


def contineous_evaluation(annotator_df, position_df, args, metric_columns):
    ''' Maps the relative judgements from the annotations to an absolute scale and
    '''
    train_df = prepare_dataset(annotator_df, position_df, keep_first_sentence=True)

    train_df = train_df.loc[train_df['worker_id'] != 'mean']

    cont_model_predictions(args, train_df, metric_columns)
    cont_model_pred_to_ann(args, train_df, metric_columns)
    cont_worker_to_worker(args, train_df)


def cont_worker_to_worker(args, train_df):
    story_ids = train_df["story_id"].unique()
    worker_story_data = []
    worker_results_data = []
    with torch.cuda.device(args["cuda_device"]):
        base_model = RelativeToAbsoluteModel()
        comparison_one_all = []
        comparison_two_all = []
        story_meta = []

        results_dict = OrderedDict()
        results_dict["measure"] = "worker"
        results_dict["training"] = "fitted"

        for story_id in story_ids:

            story_df = train_df.loc[train_df["story_id"] == story_id]
            worker_ids = story_df["worker_id"].unique()
            for worker_id, worker_id_2 in combinations(worker_ids, 2):
                with torch.no_grad():

                    meta_dict = {}

                    worker_df = story_df.loc[story_df["worker_id"] == worker_id]

                    worker_df_2 = story_df.loc[story_df["worker_id"] == worker_id_2]

                    if len(worker_df) > 0 and len(worker_df_2) > 0:

                        # meta_dict["dataset"] = "train"
                        meta_dict["worker_id"] = worker_id
                        meta_dict["worker_id_2"] = worker_id_2

                        meta_dict["training"] = "fixed"

                        suspense = torch.tensor(worker_df["suspense"].tolist()).int()
                        abs_suspense = base_model(suspense)

                        suspense_2 = torch.tensor(worker_df_2["suspense"].tolist()).int()
                        abs_suspense_2 = base_model(suspense_2)

                        if len(abs_suspense) == 0 or len(abs_suspense_2) == 0:
                            continue

                        comparison_one_all.append(abs_suspense.tolist())
                        comparison_two_all.append(abs_suspense_2.tolist())
                        story_meta.append(meta_dict)

            for story_meta_dict, predictions, annotations in zip(story_meta, comparison_one_all, comparison_two_all):
                abs_evaluate_predictions(predictions, annotations, story_meta_dict)
                worker_story_data.append(story_meta_dict)

        abs_evaluate_predictions(comparison_one_all, comparison_two_all, results_dict)

        worker_results_data.append(results_dict)

        worker_results_df = pandas.DataFrame(data=worker_results_data)
        worker_results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/worker_rel_to_abs.csv")

        worker_story_results_df = pandas.DataFrame(data=worker_story_data)
        worker_story_results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/worker_rel_to_abs_story.csv")


def cont_model_pred_to_ann(args, train_df, metric_columns):
    results_data = []
    story_data = []

    story_ids = train_df["story_id"].unique()

    with torch.cuda.device(args["cuda_device"]):

        for col in metric_columns:

            fitting_model = RelativeToAbsoluteModel()

            criterion = torch.nn.L1Loss()
            optimizer = torch.optim.SGD(fitting_model.parameters(), lr=0.01)

            for epoch in range(args["epochs"]):

                results_dict = OrderedDict()
                results_dict["measure"] = col
                results_dict["training"] = "fitted"
                # results_dict["dataset"] = "train"

                comparison_one_all = []
                comparison_two_all = []
                story_meta = []

                story_ids = train_df["story_id"].unique()

                for story_id in story_ids:

                    story_df = train_df.loc[train_df["story_id"] == story_id]
                    worker_ids = story_df["worker_id"].unique()
                    for worker_id in worker_ids:

                        if worker_id in ["median", "mean"]:
                            continue
                            # Only compare against the real annotators not the virtual average.

                        meta_dict = {}

                        worker_df = story_df.loc[story_df["worker_id"] == worker_id]

                        if len(worker_df) > 0:

                            # meta_dict["dataset"] = "train"
                            meta_dict["story_id"] = story_id
                            meta_dict["worker_id"] = worker_id
                            meta_dict["measure"] = col

                            if epoch == 0:
                                meta_dict["training"] = "fixed"
                            else:
                                meta_dict["training"] = "fitted"

                            suspense_list = worker_df["suspense"].tolist()
                            measure_values = worker_df[f"{col}_scaled"].tolist()
                            measure_values_unscaled = worker_df[f"{col}"].tolist()

                            if col != "baseclass" and "sentiment" not in col and sum(
                                    [1 for i in measure_values_unscaled if i > 0.0]) > 0:
                                # print(measure_values, measure_values_unscaled, suspense_list)
                                measure_values, measure_values_unscaled, suspense_list = zip(
                                    *((m, mu, s) for m, mu, s in
                                      zip(measure_values,
                                          measure_values_unscaled,
                                          suspense_list) if
                                      mu > 0.0))

                            suspense = torch.tensor(suspense_list).int()
                            model_predictions = torch.tensor(measure_values)

                            # print(suspense, len(suspense), model_predictions, len(model_predictions))

                            measure_offset = 0.0 - model_predictions[0]
                            model_predictions = torch.tensor([m + measure_offset for m in model_predictions],
                                                             requires_grad=True)

                            abs_suspense = fitting_model(suspense)

                            if len(model_predictions) == 0 or len(abs_suspense) == 0:
                                continue

                            model_predictions_list = model_predictions.tolist()
                            abs_suspense_list = abs_suspense.tolist()

                            model_predictions_list = model_predictions_list[
                                                     0: min(len(model_predictions_list), len(abs_suspense_list))]
                            abs_suspense_list = abs_suspense_list[
                                                0: min(len(model_predictions_list), len(abs_suspense_list))]

                            # print(model_predictions_list, abs_suspense_list)
                            if len(model_predictions_list) >= 2:
                                comparison_one_all.append(model_predictions_list)
                                comparison_two_all.append(abs_suspense_list)
                                story_meta.append(meta_dict)

                            if abs_suspense.size(0) == model_predictions.size(0):

                                loss = criterion(abs_suspense, model_predictions)

                                if epoch > 0 and col != "baseclass":
                                    optimizer.zero_grad()
                                    loss.backward()
                                    optimizer.step()

                if epoch == 0 or epoch == args["epochs"] - 1:
                    if epoch == 0:
                        results_dict["training"] = "fixed"

                    for story_meta_dict, predictions, annotations in zip(story_meta, comparison_one_all,
                                                                         comparison_two_all):
                        abs_evaluate_predictions(predictions, annotations, story_meta_dict)
                        story_data.append(story_meta_dict)

                    # print(comparison_one_all, comparison_two_all, results_dict)
                    abs_evaluate_predictions(comparison_one_all, comparison_two_all, results_dict)

                    results_data.append(results_dict)
        results_df = pandas.DataFrame(data=results_data)
        results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/model_to_ann_rel_to_abs.csv")

        story_results_df = pandas.DataFrame(data=story_data)
        story_results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/model_to_ann_rel_to_abs_story.csv")
    return col, story_ids


def cont_model_predictions(args, train_df, metric_columns):
    prediction_story_data = []
    prediction_results_data = []
    for col, col_2 in combinations(metric_columns, 2):

        comparison_one_all = []
        comparison_two_all = []
        story_meta = []

        results_dict = OrderedDict()
        results_dict["training"] = "fixed"
        results_dict["measure"] = col
        results_dict["measure_2"] = col_2

        story_ids = train_df["story_id"].unique()

        for story_id in story_ids:

            story_df = train_df.loc[train_df["story_id"] == story_id]

            meta_dict = {}

            meta_dict["story_id"] = story_id
            meta_dict["measure"] = col
            meta_dict["measure_2"] = col_2
            meta_dict["training"] = "fixed"

            model_predictions = story_df[f"{col}_scaled"].tolist()
            model_predictions_2 = story_df[f"{col_2}_scaled"].tolist()

            if len(model_predictions) == 0 or len(model_predictions_2) == 0:
                continue

            comparison_one_all.append(model_predictions)
            comparison_two_all.append(model_predictions_2)

            story_meta.append(meta_dict)

        for story_meta_dict, predictions, annotations in zip(story_meta, comparison_one_all, comparison_two_all):
            abs_evaluate_predictions(predictions, annotations, story_meta_dict)
            prediction_story_data.append(story_meta_dict)

        abs_evaluate_predictions(comparison_one_all, comparison_two_all, results_dict)

        prediction_results_data.append(results_dict)
    prediction_results_df = pandas.DataFrame(data=prediction_results_data)
    prediction_results_df.to_csv(f"{args['output_dir']}/sentence_model_evaluation/prediction_rel_to_abs.csv")
    prediction_story_results_df = pandas.DataFrame(data=prediction_story_data)
    prediction_story_results_df.to_csv(
        f"{args['output_dir']}/sentence_model_evaluation/prediction_rel_to_abs_story.csv")

def abs_evaluate_predictions(predictions, annotations, results_dict):
    if any(isinstance(el, (list, tuple, Iterable)) for el in annotations):

        kendall_list = []
        spearman_list = []
        pearson_list = []

        kendall_pvalue_list = []
        spearman_pvalue_list = []
        pearson_pvalue_list = []

        l1_distance_list = []
        l2_distance_list = []

        for pred, ann in zip(predictions, annotations):
            try:

                if len(pred) != len(ann):
                    # print("List not equal", results_dict, pred, ann)
                    continue
                else:
                    pass  # print(results_dict, pred, ann)

                pearson, pvalue = pearsonr(pred, ann)
                pearson_list.append(pearson)
                pearson_pvalue_list.append(pvalue)
                kendall, pvalue = kendalltau(pred, ann, nan_policy="omit")
                kendall_list.append(kendall)
                kendall_pvalue_list.append(pvalue)
                spearman, pvalue = spearmanr(pred, ann)
                spearman_list.append(spearman)
                spearman_pvalue_list.append(pvalue)

                l2_distance_list.append(distance.euclidean(pred, ann) / len(ann))
                l1_distance_list.append(distance.cityblock(pred, ann) / len(ann))

            except Exception as ex:
                print(ex)

        def ci(r, n, alpha=0.05):

            r_z = numpy.arctanh(r)
            se = 1 / numpy.sqrt(n - 3)
            z = scipy.stats.norm.ppf(1 - alpha / 2)
            lo_z, hi_z = r_z - z * se, r_z + z * se
            lo, hi = numpy.tanh((lo_z, hi_z))
            return lo, hi

        try:
            results_dict[f"pearson"] = sum(pearson_list) / float(len(pearson_list))
            results_dict[f"kendall"] = sum(kendall_list) / float(len(kendall_list))
            results_dict[f"spearman"] = sum(spearman_list) / float(len(spearman_list))

            results_dict[f"pearson_agg_stat"], results_dict[f"pearson_pvalue"] = combine_pvalues(
                [x for x in pearson_pvalue_list if x > 0.0 and x < 1.0], method="mudholkar_george")
            results_dict[f"kendall_agg_stat"], results_dict[f"kendall_pvalue"] = combine_pvalues(
                [x for x in kendall_pvalue_list if x > 0.0 and x < 1.0], method="mudholkar_george")
            results_dict[f"spearman_agg_stat"], results_dict[f"spearman_pvalue"] = combine_pvalues(
                [x for x in spearman_pvalue_list if x > 0.0 and x < 1.0], method="mudholkar_george")

            results_dict[f"pearson_low_95"], results_dict[f"pearson_high_95"] = ci(results_dict[f"pearson"],
                                                                                   len(pearson_list))
            results_dict[f"kendall_low_95"], results_dict[f"kendall_high_95"] = ci(results_dict[f"kendall"],
                                                                                   len(kendall_list))
            results_dict[f"spearman_low_95"], results_dict[f"spearman_high_95"] = ci(results_dict[f"spearman"],
                                                                                     len(spearman_list))

            results_dict[f"pearson_low_99"], results_dict[f"pearson_high_99"] = ci(results_dict[f"pearson"],
                                                                                   len(pearson_list), alpha=0.01)
            results_dict[f"kendall_low_99"], results_dict[f"kendall_high_99"] = ci(results_dict[f"kendall"],
                                                                                   len(kendall_list), alpha=0.01)
            results_dict[f"spearman_low_99"], results_dict[f"spearman_high_99"] = ci(results_dict[f"spearman"],
                                                                                     len(spearman_list), alpha=0.01)

            results_dict[f"l2_distance"] = sum(l2_distance_list) / float(len(l2_distance_list))
            results_dict[f"l1_distance"] = sum(l1_distance_list) / float(len(l1_distance_list))

        except Exception as ex:
            print(ex)
        # results_dict[f"alpha"] = sum(alpha_list) / float(len(annotations))

    else:

        if len(predictions) != len(annotations):
            # print("List not equal", results_dict, predictions, annotations)
            return
        else:
            pass  # print(results_dict, predictions, annotations)

        try:

            results_dict[f"pearson"], results_dict[f"pearson_pvalue"] = pearsonr(predictions,
                                                                                 annotations)
            results_dict[f"kendall"], results_dict[f"kendall_pvalue"] = kendalltau(
                predictions, annotations, nan_policy="omit")
            results_dict[f"spearman"], results_dict[f"spearman_pvalue"] = spearmanr(
                predictions, annotations)

            results_dict["l2_distance"] = distance.euclidean(predictions, annotations) / len(annotations)
            results_dict["l1_distance"] = distance.cityblock(predictions, annotations) / len(annotations)

            print(results_dict, predictions, annotations)

        except Exception as ex:
            print(ex)


def plot_annotator_and_model_predictions(position_df, annotator_df, args, metric_columns,
                                         model=RelativeToAbsoluteModel()):
    print(f"Plot the annotator sentences to get a visualisation of the peaks in the annotations.")

    if args["export_only"]:
        columns = args["export_columns"]
    else:
        columns = metric_columns

    position_df = scale_prediction_columns(position_df)

    colors = plotly.colors.DEFAULT_PLOTLY_COLORS

    story_ids = annotator_df["story_id"].unique()

    position_story_ids = position_df["story_id"].unique()

    story_ids = set(story_ids).union(set(position_story_ids))

    with torch.no_grad():

        for story_id in story_ids:

            position_story_df = position_df.loc[position_df["story_id"] == story_id]
            if len(position_story_df) > 0:
                sentence_text = position_story_df["text"].tolist()
                text = sentence_text
            else:
                text = None

            plot_data = []

            story_df = annotator_df.loc[annotator_df["story_id"] == story_id]
            story_df = story_df.groupby(['story_id', 'sentence_num', 'worker_id'],
                                        as_index=False).first()

            if len(story_df) > 0:

                worker_ids = set(story_df["worker_id"].unique())

                for worker_id in worker_ids:

                    if worker_id == "mean":
                        continue

                    worker_df = story_df.loc[story_df["worker_id"] == worker_id]

                    if len(worker_df) > 0:

                        worker_df = worker_df.sort_values(by=["sentence_num"])

                        suspense = torch.tensor(worker_df["suspense"].tolist()).int()

                        measure_values = model(suspense).tolist()

                        dash = "solid"
                        if worker_id == "median":
                            dash = "dash"

                        trace = go.Scatter(
                            x=worker_df["sentence_num"],
                            y=measure_values,
                            mode='lines+markers',
                            name=f"{worker_id}",
                            line=dict(color=colors[7], dash=dash)
                        )
                        plot_data.append(trace)

                plot_data.append(trace)

            story_df = position_df.loc[position_df["story_id"] == story_id]

            if len(story_df) > 0:

                for i, col in enumerate(columns):
                    print(position_df.columns)
                    if col in list(position_df.columns):

                        measure_values = position_df[f"{col}_scaled"].tolist()
                        measure_values_unscaled = position_df[f"{col}"].tolist()
                        sentence_nums = worker_df["sentence_num"].tolist()

                        if len(measure_values_unscaled) == 0 or len(measure_values) == 0 or len(sentence_nums) == 0:
                            continue


                        if col != "baseclass" and "sentiment" not in col and sum(
                                [1 for i in measure_values_unscaled if i > 0]) > 0:
                            measure_values, measure_values_unscaled, sentence_nums = zip(
                                *((m, mu, s) for m, mu, s in zip(measure_values, measure_values_unscaled, sentence_nums) if
                                  mu > 0.0))

                        measure_offset = 0.0 - measure_values[0]
                        measure_values = [m + measure_offset for m in measure_values]

                        trace = go.Scatter(
                            x=sentence_nums,
                            y=measure_values,
                            mode='lines+markers',
                            name=f"{col}",
                            text=text,
                            color=i,
                            colorscale='Viridis',
                        )
                        plot_data.append(trace)

            title = f'Model and Annotation plots {story_id}'
            if args["export_only"]:
                title = None

            if len(plot_data) > 0:
                layout = go.Layout(
                    title=title,
                    hovermode='closest',
                    xaxis=dict(
                        title=f"Sentence",
                    ),
                    yaxis=dict(
                        title=f"Suspense",
                    ),
                    showlegend=not args["export_only"],
                    legend=dict(
                        orientation="h")
                )

                fig = go.Figure(data=plot_data, layout=layout)

                export_plots(args, f"/model_annotation_plots/{story_id}", fig)


def extract_rows(args):
    index_counter = 0
    with jsonlines.open(args['prediction_json']) as reader:
        for i, obj in tqdm(enumerate(reader)):
            story_id = obj["story_id"]

            processed_dict_list = []
            for child in obj["sentences"]:
                processed_dict = {}
                processed_dict["story_id"] = story_id
                processed_dict["text"] = child["text"]
                processed_dict["sentence_num"] = child["sentence_num"]
                processed_dict["index"] = index_counter

                if "prediction_metrics" in child:
                    metrics = child["prediction_metrics"]
                    for k_level, v_level in metrics.items():
                        for k_metric, v_metric in v_level.items():
                            processed_dict[f"metric.{k_level}.{k_metric}"] = v_metric

                processed_dict_list.append(processed_dict)

            index_counter += 1

            df = pandas.read_json(StringIO(json.dumps(processed_dict_list)))
            yield df


def evaluate_stories(args):
    print(f"Evaluate stories: {args}")

    ensure_dir(f"{args['output_dir']}/sentence_model_evaluation/")

    df_list = [r for r in extract_rows(args)]

    position_df = pandas.concat(df_list)

    metric_columns = [i for i in list(position_df.columns) if i.startswith("metric")]
    print(metric_columns)

    position_df["baseclass"] = 0.0
    position_df["random"] = numpy.random.randint(1, 100, position_df.shape[0])

    print(f"Position rows : {len(position_df)}")
    annotator_df = pd.read_csv(args["annotator_targets"])

    if args["exclude_worker_ids"] is not None and len(args["exclude_worker_ids"]) > 0:
        annotator_df = annotator_df[~annotator_df["worker_id"].isin(args["exclude_worker_ids"])]

    plot_annotator_and_model_predictions(position_df, annotator_df, args, metric_columns)
    if annotator_df:
        contineous_evaluation(position_df, annotator_df, args, metric_columns)


def scale_prediction_columns(position_df):
    for col in metric_columns:
        if col not in position_df.columns:
            continue

        scaler = StandardScaler()
        scaled_col = numpy.squeeze(scaler.fit_transform(position_df[col].to_numpy().reshape(-1, 1)),
                                   axis=1).tolist()
        position_df[f"{col}_scaled"] = scaled_col
    return position_df


def prepare_dataset(annotator_df, position_df, keep_first_sentence=False):
    merged_df = pd.merge(position_df, annotator_df, left_on=["story_id", "sentence_num"],
                         right_on=["story_id", "sentence_num"], how="inner")
    merged_df = merged_df.sort_values(by=["worker_id", "story_id", "sentence_num"]).reset_index()

    merged_df = pd.concat([merged_df, merged_df[1:].reset_index(drop=True).add_suffix("_later")],
                          axis=1)
    merged_df = merged_df.sort_values(by=["worker_id", "story_id", "sentence_num"]).reset_index()
    merged_df = merged_df.loc[merged_df["story_id"] == merged_df["story_id_later"]]
    merged_df = merged_df.loc[merged_df["worker_id"] == merged_df["worker_id_later"]]

    if keep_first_sentence:
        merged_df = merged_df.loc[(merged_df["suspense"] != 0.0) | (merged_df["sentence_num"] == 0)]
    else:
        merged_df = merged_df.loc[merged_df["suspense"] != 0.0]

    max_df = merged_df.groupby(by=["story_id"], as_index=False)["sentence_num"].max()
    df_all = merged_df.merge(max_df, on=['story_id', 'sentence_num'],
                             how='left', indicator=True)
    merged_df = df_all.loc[df_all['_merge'] == "left_only"]

    merged_df = scale_prediction_columns(merged_df)
    print(f"Merged rows: {len(merged_df)}")

    merged_df = merged_df.sort_values(by=["story_id", "worker_id", "sentence_num"])

    return merged_df


def export_plots(args, file, fig):
    ensure_dir(f"{args['output_dir']}/{file}")
    if not args["no_html_plots"]:
        file_path = f"{args['output_dir']}/{file}.html"
        print(f"Save plot: {file_path}")
        pio.write_html(fig, file_path)

    if not args["no_pdf_plots"]:
        file_path = f"{args['output_dir']}/{file}.pdf"
        print(f"Save plot pdf: {file_path}")
        pio.write_image(fig, file_path)

evaluate_stories(vars(args))