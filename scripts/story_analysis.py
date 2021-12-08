''' Create Analysis Charts for Stories in bulk based on the preidction output and cluster analysis.

'''
import argparse
import os
import statistics
from copy import deepcopy
from itertools import zip_longest
from math import floor, ceil
from textwrap import TextWrapper
from io import StringIO
from itertools import combinations
import json

import colorlover as cl
import numpy
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
# These are the default plotly colours.
from scipy.signal import find_peaks
from scipy.stats import kendalltau, spearmanr, pearsonr
from sklearn.preprocessing import RobustScaler
from jsonlines import jsonlines
from tqdm import tqdm
import pandas

colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
          'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
          'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
          'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
          'rgb(188, 189, 34)', 'rgb(23, 190, 207)']

shapes = ["circle", "square", "diamond", "cross", "x", "star-triangle-up", "star-triangle-down", "triangle-up",
          "triangle-down", "triangle-left", "triangle-right", "pentagon", "hexagon", "octagon", 'hexagram', "bowtie",
          "hourglass"]

parser = argparse.ArgumentParser(
    description='Run stats from  the prediction output, clustering and stats for the annotations and predictions.')
parser.add_argument('--prediction-json', required=False, type=str, help="CSV of the prediction position stats.")
parser.add_argument('--output-dir', required=True, type=str, help="CSV containing the vector output.")
parser.add_argument('--smoothing', required=False, type=str, nargs='*',
                    default=['exp', 'holt', 'avg', 'avg_2', 'reg', 'reg_2', 'arima'],
                    help="CSV containing the vector output.")
parser.add_argument('--max-plot-points', default=50000, type=int, help="Max number of scatter points.")
parser.add_argument('--cluster-example-num', default=100, type=int,
                    help="Max number of examples to select for each cluster category.")
parser.add_argument("--smoothing-plots", default=False, action="store_true",
                    help="Plot sliding windows and smoothong as well as the raw position data.")
parser.add_argument("--no-html-plots", default=False, action="store_true", help="Don't save plots to HTML")
parser.add_argument("--no-pdf-plots", default=False, action="store_true", help="Don't save plots to PDF")
parser.add_argument("--no-cluster-output", default=False, action="store_true",
                    help="Don't calculate the cluster output.")
parser.add_argument("--no-story-output", default=False, action="store_true", help="Don't calculate the story plots.")
parser.add_argument('--peak-prominence-weighting', default=0.35, type=float,
                    help="Use to scale the standard deviation of a column.")
parser.add_argument('--peak-width', default=1.0, type=float,
                    help="How wide must a peak be to be included. 1.0 allow a single point sentence to be a peak.")
parser.add_argument('--number-of-peaks', default=-1, type=int,
                    help="Number of peaks to find, overrides the other settings.")
parser.add_argument('--turning-points-csv', required=False, type=str,
                    help="If provided the turning points to compare against from the CSV.")
parser.add_argument('--turning-point-columns', required=False, type=str, nargs="*",
                    default=["tp1", "tp2", "tp3", "tp4", "tp5"],
                    help="If provided the turning points to compare against from the CSV.")
parser.add_argument('--turning-point-means', required=False, type=float, nargs="*",
                    default=[11.39, 31.86, 50.65, 74.15, 89.43],
                    help="If turning points provided then these are the expected positions.")
parser.add_argument('--turning-point-stds', required=False, type=float, nargs="*",
                    default=[6.72, 11.26, 12.15, 8.40, 4.74],
                    help="If turning points provided then these are the expected positions.")
parser.add_argument("--export-only", default=False, action="store_true", help="Only for export so remove legend and filter down export parameters.")
parser.add_argument('--export-columns', required=False, type=str, nargs="*",
                    default=[])


args = parser.parse_args()


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        print(f"Create directory: {directory}")
        os.makedirs(directory)


def scale_prediction_columns(position_df,prediction_columns):
    for col in prediction_columns:

        if not col in position_df.columns:
            continue

        scaler = RobustScaler()
        scaled_col = numpy.squeeze(scaler.fit_transform(position_df[col].to_numpy().reshape(-1, 1)), axis=1).tolist()
        position_df[f"{col}_scaled"] = scaled_col
        print(position_df[f"{col}_scaled"], scaled_col)

    #position_df = position_df.reset_index()

    return position_df

def create_story_plots(position_df,turning_points_df=None,args=None):

    ensure_dir(f"{args['output_dir']}/prediction_plots/")

    export_columns = args["export_columns"]



    #position_df = position_df.sort_values(by=["story_id", "sentence_num"]).reset_index()


    # merged_df = merged_df.loc[merged_df["sentence_num"] + 1 == merged_df["sentence_num_later"]]

    prediction_columns = [c for c in position_df.columns if "metric" in c]
    position_df = scale_prediction_columns(position_df,prediction_columns=prediction_columns)
    prediction_columns = [c for c in position_df.columns if "_scaled" in c]

    #print(position_df.columns)

    position_df = position_df.fillna(value=0.0)


    segmented_data = []

    column_list = []

    columns = prediction_columns
    if export_columns:
        columns = export_columns

    for i, pred in enumerate(columns):


            column_list.append(f"{pred}_scaled")


    turning_point_data_list = []

    story_ids = position_df.groupby("story_id")

    for story_id, group in story_ids:

        group_df = group.sort_values(by=['sentence_num'])

        print(group_df)


        plotted_turning_points = False

        data = []

        prom_data = []


        prominence_threshold = 0.00001  # Make tiny as is not specified then the metadata is not returned for prominences.

        color_idx = 0

        max_point = 0.0
        for i, pred in enumerate(prediction_columns):

            pred_data = group_df[pred].tolist()

            if len(pred_data) == 0:
                continue

            sentence_nums = group_df["sentence_num"].tolist()
            sentence_text = group_df["text"].tolist()

            if args["export_only"]:
               pred_data = pred_data[:len(pred_data) - 1]
               sentence_nums = sentence_nums[:len(sentence_nums) - 1]
               sentence_text = sentence_text[:len(sentence_text) - 1]

            measure_offset = 0.0 - pred_data[0]
            pred_data = [m + measure_offset for m in pred_data]

            pred_name = pred.replace('_scaled','')


            text = [f"<b>{t}</b>" for t in group_df["text"]]


            max_point = max(max_point, max(group_df[pred]))

            trace = go.Scatter(
                x=sentence_nums,
                y=pred_data,
                text=text,
                mode='lines+markers',
                line=dict(
                    color=colors[color_idx % len(colors)]
                ),
                name=f'{pred_name}',
            )
            data.append(trace)

            y = pred_data

            type = "peak"
            peak_indices, peaks_meta = find_peaks(y, prominence=prominence_threshold, width=args["peak_width"],
                                                  plateau_size=1)

            num_of_peaks = args["number_of_peaks"]
            peak_indices = optionally_top_n_peaks(num_of_peaks, peak_indices, peaks_meta)

            if len(peak_indices) > 0:
                hover_text, peaks_data = create_peak_text_and_metadata(peak_indices, peaks_meta, sentence_nums,
                                                                       sentence_text, story_id, "peak", pred)

                segmented_data.extend(peaks_data)

                trace = go.Scatter(
                    x=[sentence_nums[j] for j in peak_indices],
                    y=[y[j] for j in peak_indices],
                    mode='markers',
                    marker=dict(
                        color=colors[color_idx % len(colors)],
                        symbol='star-triangle-up',
                        size=11,
                    ),
                    name=f'{pred_name} - {type}',
                    text=hover_text
                )
                data.append(trace)

            type = "trough"
            y_inverted = [x_n * -1.0 for x_n in y]
            # prominence=prom, width=args["peak_width"]
            if len(peak_indices) > 0:
                peak_indices, peaks_meta = find_peaks(y_inverted, prominence=prominence_threshold,
                                                      width=args["peak_width"], plateau_size=1)

                peak_indices = optionally_top_n_peaks(num_of_peaks, peak_indices, peaks_meta)

                hover_text, peaks_data = create_peak_text_and_metadata(peak_indices, peaks_meta, sentence_nums,
                                                                       sentence_text, story_id, "trough", pred)

                segmented_data.extend(peaks_data)

                trace = go.Scatter(
                    x=[sentence_nums[j] for j in peak_indices],
                    y=[y[j] for j in peak_indices],
                    mode='markers',
                    marker=dict(
                        color=colors[color_idx % len(colors)],
                        symbol='star-triangle-down',
                        size=11,
                    ),
                    name=f'{pred_name} - {type}',
                    text=hover_text
                )
                data.append(trace)

            all_points = []
            if turning_points_df is not None:

                turning_story_df = turning_points_df.loc[turning_points_df["story_id"] == story_id]
                if len(turning_story_df) > 0:

                    all_points = []

                    story_length = len(sentence_nums)
                    expected_points_indices = []

                    mean_expected = []
                    lower_brackets = []
                    upper_brackets = []
                    for mean, std in zip(args["turning_point_means"], args["turning_point_stds"]):
                        mean_pos = int(round(mean * (story_length / 100)))
                        mean_expected.append(mean_pos)
                        lower_bracket = max(0, int(floor(mean - std) * (story_length / 100)))
                        lower_brackets.append(lower_bracket)
                        upper_bracket = min(story_length - 1, int(ceil(mean + std) * (story_length / 100)))
                        upper_brackets.append(upper_bracket)

                        sentence_whole = group_df[pred].tolist()

                        index_pos = sentence_whole.index(max(sentence_whole[lower_bracket:upper_bracket + 1]))
                        expected_points_indices.append(index_pos)

                    done_story = False
                    for i, ann_point in turning_story_df.iterrows():

                        if done_story:
                            continue

                        turn_dict = {}

                        turn_dict["story_id"] = story_id
                        turn_dict["measure"] = pred
                        turn_dict["annotator"] = i

                        turn_dict["lower_brackets"] = lower_brackets
                        turn_dict["upper_brackets"] = upper_brackets
                        turn_dict["mean_expected"] = mean_expected

                        annotated_points = []
                        for col in args["turning_point_columns"]:
                            point = ann_point[col]
                            annotated_points.append(point)

                        all_points.extend(annotated_points)

                        if len(expected_points_indices) == len(args["turning_point_columns"]):
                            exp_dict = deepcopy(turn_dict)

                            calc_turning_point_distances(annotated_points, args, expected_points_indices,
                                                         sentence_nums,
                                                         exp_dict, type="constrained", compared="annotated")
                            turning_point_data_list.append(exp_dict)

                        if len(peak_indices) == len(args["turning_point_columns"]):
                            peak_dict = deepcopy(turn_dict)
                            calc_turning_point_distances(annotated_points, args, peak_indices, sentence_nums,
                                                         peak_dict, type="unconstrained", compared="annotated")
                            turning_point_data_list.append(peak_dict)

                    if len(peak_indices) == len(args["turning_point_columns"]):
                        exp_dict = deepcopy(turn_dict)
                        calc_turning_point_distances(mean_expected, args, expected_points_indices, sentence_nums,
                                                     exp_dict, type="constrained", compared="dist_baseline")
                        turning_point_data_list.append(exp_dict)

                    if len(peak_indices) == len(args["turning_point_columns"]):
                        peak_dict = deepcopy(turn_dict)
                        calc_turning_point_distances(mean_expected, args, peak_indices, sentence_nums,
                                                     peak_dict, type="unconstrained", compared="dist_baseline")
                        turning_point_data_list.append(peak_dict)

                if len(expected_points_indices) > 0:
                    trace = go.Scatter(
                        x=[sentence_nums[j] for j in expected_points_indices if j < len(sentence_nums)],
                        y=[y[j] for j in expected_points_indices if j < len(y)],
                        mode='markers',
                        marker=dict(
                            color=colors[color_idx % len(colors)],
                            symbol='triangle-up',
                            size=11,
                        ),
                        name=f'{pred_name} - {type} constrained',
                        text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in expected_points_indices if j < len(sentence_nums)],
                    )
                    data.append(trace)

                if len(annotated_points) > 0 and not plotted_turning_points:
                    plotted_turning_points = True

                    trace = go.Scatter(
                        x=[sentence_nums[p] for p in mean_expected if p < len(sentence_nums)],
                        y=[0.0] * len(mean_expected),
                        mode='markers',
                        marker=dict(
                            color="black",
                            symbol='diamond',
                            size=11,
                        ),
                        name=f'dist baseline',
                        text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in mean_expected if
                              j < len(sentence_nums)],

                    )
                    data.append(trace)

                    if not args["export_only"]:

                        trace = go.Scatter(
                            x=[sentence_nums[p] for p in lower_brackets if p < len(sentence_nums)],
                            y=[0.0] * len(lower_brackets),
                            mode='markers',
                            marker=dict(
                                color="black",
                                symbol='triangle-right',
                                size=11,
                            ),
                            name=f'dist baseline lower',
                            text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in lower_brackets if
                                  j < len(sentence_nums)],

                        )
                        data.append(trace)

                        trace = go.Scatter(
                            x=[sentence_nums[p] for p in upper_brackets if p < len(sentence_nums)],
                            y=[0.0] * len(upper_brackets),
                            mode='markers',
                            marker=dict(
                                color="black",
                                symbol='triangle-left',
                                size=11,
                            ),
                            name=f'dist baseline upper',
                            text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in upper_brackets if
                                  j < len(sentence_nums)],

                        )
                        data.append(trace)

                    trace = go.Scatter(
                        x=[sentence_nums[p] for p in all_points if p < len(sentence_nums)],
                        y=[0.0] * len(all_points),
                        mode='markers',
                        marker=dict(
                            color="gold",
                            symbol='star',
                            size=11,
                        ),
                        name=f'annotated',
                        text=[f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>" for j in all_points if
                              j < len(sentence_nums)],

                    )
                    data.append(trace)

            color_idx += 1

        title = f'Story {story_id} Prediction Plot'
        if args["export_only"]:
            title=None

        layout = go.Layout(
            title=title,
            hovermode='closest',
            xaxis=dict(
                title='Sentence',
            ),
            yaxis=dict(
                title=f'Suspense',
            ),
            showlegend=not args["export_only"],
            legend=dict(
                orientation="h")
        )

        fig = go.Figure(data=data, layout=layout)

        if not args["no_html_plots"]:
            file_path = f"{args['output_dir']}/prediction_plots/story_{story_id}_plot.html"
            print(f"Save plot {file_path}")
            pio.write_html(fig, file_path, include_plotlyjs='cdn', include_mathjax='cdn', auto_open=False)
        if not args["no_pdf_plots"]:
            file_path = f"{args['output_dir']}/prediction_plots/story_{story_id}_plot.pdf"
            print(f"Save plot pdf: {file_path}")
            pio.write_image(fig, file_path)

    segmented_data = pd.DataFrame(data=segmented_data)
    segmented_data.to_csv(f"{args['output_dir']}/prediction_plots/peaks_and_troughs.csv")

    if len(turning_point_data_list) > 0:
        ensure_dir(f"{args['output_dir']}/turning_points/")
        turning_point_eval_df = pd.DataFrame(data=turning_point_data_list)
        turning_point_eval_df = turning_point_eval_df.fillna(value=0.0)

        summary_data_list = []

        for group_by, group in turning_point_eval_df.groupby(by=["measure", "constraint_type", "compared"]):

            measure, constraint_type, compared = group_by

            summary_dict = {}

            summary_dict["measure"] = measure
            summary_dict["constraint_type"] = constraint_type
            summary_dict["compared"] = compared
            summary_dict["num_of_stories_counted"] = len(group)

            summary_dict["total_agreement"] = group["total_agreement"].mean()
            summary_dict["partial_agreement"] = group["partial_agreement"].sum() / len(group["partial_agreement"])

            predict_all = []
            actual_all = []
            for col in args["turning_point_columns"]:

                summary_dict[f"{col}_total_agreement"] = group[f"{col}_total_agreement_correct"].sum() / len(
                    group[f"{col}_total_agreement_correct"])

                predicted_positions = group[f"{col}_predicted_relative_position"].tolist()
                predict_all.extend(predicted_positions)
                actual_positions = group[f"{col}_expected_relative_position"].tolist()
                actual_all.extend(actual_positions)

                if len(predicted_positions) >= 2 and len(actual_positions) >= 2:

                    #print("Correlation", predicted_positions, actual_positions)

                    kendall, kendall_p_value = kendalltau(predicted_positions, actual_positions)
                    spearman, spearman_p_value = spearmanr(predicted_positions, actual_positions)
                    pearson, pearson_p_value = pearsonr(predicted_positions, actual_positions)

                    summary_dict[f"{col}_predicted_relative_position_corr_kendall"] = kendall
                    summary_dict[f"{col}_predicted_relative_position_corr_kendall_p_value"] = kendall_p_value
                    summary_dict[f"{col}_predicted_relative_position_corr_spearman"] = spearman
                    summary_dict[f"{col}_predicted_relative_position_corr_spearman_p_value"] = spearman_p_value
                    summary_dict[f"{col}_predicted_relative_position_corr_pearson"] = pearson
                    summary_dict[f"{col}_predicted_relative_position_corr_pearson_p_value"] = pearson_p_value

            if len(predicted_positions) >= 2 and len(actual_positions) >= 2:
                
                print("Correlation", predicted_positions, actual_positions)

                kendall, kendall_p_value = kendalltau(predict_all, actual_all)
                spearman, spearman_p_value = spearmanr(predict_all, actual_all)
                pearson, pearson_p_value = pearsonr(predict_all, actual_all)

                summary_dict[f"predicted_relative_position_all_corr_kendall"] = kendall
                summary_dict[f"predicted_relative_position_all_corr_kendall_p_value"] = kendall_p_value
                summary_dict[f"predicted_relative_position_all_corr_spearman"] = spearman
                summary_dict[f"predicted_relative_position_all_corr_spearman_p_value"] = spearman_p_value
                summary_dict[f"predicted_relative_position_all_corr_pearson"] = pearson
                summary_dict[f"predicted_relative_position_all_corr_pearson_p_value"] = pearson_p_value

            for c in args["turning_point_columns"] + ["avg"]:

                for d in ["dist", "norm_dist", "abs_dist", "abs_norm_dist"]:
                    col_series = group[f"{c}_{d}"]

                    col_stats = col_series.describe()

                    summary_dict[f"{c}_{d}_mean"] = col_stats["mean"]
                    summary_dict[f"{c}_{d}_std"] = col_stats["std"]
                    summary_dict[f"{c}_{d}_min"] = col_stats["min"]
                    summary_dict[f"{c}_{d}_max"] = col_stats["max"]
                    summary_dict[f"{c}_{d}_25"] = col_stats["25%"]
                    summary_dict[f"{c}_{d}_50"] = col_stats["50%"]
                    summary_dict[f"{c}_{d}_75"] = col_stats["75%"]

            summary_data_list.append(summary_dict)

        turning_point_eval_df.to_csv(f"{args['output_dir']}/turning_points/turning_point_eval_all.csv")
        turning_point_summary_df = pd.DataFrame(data=summary_data_list)

        turning_point_summary_df.to_csv(f"{args['output_dir']}/turning_points/summary_evaluation.csv")

        # Calculate summary stats.


def calc_turning_point_distances(annotated_points, args, peak_indices, sentence_nums, turn_dict, type="unconstrained",
                                 compared="annotated"):
    turn_dict["constraint_type"] = type
    turn_dict["compared"] = compared

    num_of_sentences = max(sentence_nums)

    for col, p in zip(args["turning_point_columns"], peak_indices):
        turn_dict[f"{col}_predicted_relative_position"] = p / len(sentence_nums)
        turn_dict[f"{col}_predicted_position"] = p

    for col, p in zip(args["turning_point_columns"], annotated_points):
        turn_dict[f"{col}_expected_relative_position"] = p / len(sentence_nums)
        turn_dict[f"{col}_expected_position"] = p

    points_set = set(annotated_points)
    peak_indices_set = set(peak_indices)

    for col, pred, exp in zip(args["turning_point_columns"], peak_indices, annotated_points):
        turn_dict[f"{col}_total_agreement_correct"] = int(pred == exp)

    turn_dict["total_agreement"] = len(peak_indices_set.intersection(points_set)) / len(points_set)
    turn_dict["partial_agreement"] = int(len(peak_indices_set.intersection(points_set)) > 0)
    turn_dict["annotated_total"] = len(points_set)

    distances = []
    norm_distances = []
    abs_distances = []
    abs_norm_distances = []

    for predicted, actual in zip_longest(peak_indices, annotated_points, fillvalue=peak_indices[-1]):
        distance = predicted - actual
        distances.append(distance)

        norm_distances.append(distance / float(num_of_sentences))
        abs_distance = abs(distance)
        abs_distances.append(abs_distance)
        abs_norm_distances.append(abs_distance / float(num_of_sentences))

    turn_dict[f"avg_abs_norm_dist"] = sum(abs_norm_distances) / len(abs_norm_distances)
    turn_dict[f"avg_abs_dist"] = sum(abs_distances) / len(abs_distances)
    turn_dict[f"avg_dist"] = sum(distances) / len(distances)
    turn_dict[f"avg_norm_dist"] = sum(norm_distances) / len(norm_distances)

    for norm, abs_dist, norm_dist, dist, col in zip(abs_norm_distances, abs_distances, norm_distances, distances,
                                                    args["turning_point_columns"]):
        turn_dict[f"{col}_abs_norm_dist"] = norm
        turn_dict[f"{col}_abs_dist"] = abs_dist
        turn_dict[f"{col}_norm_dist"] = norm_dist
        turn_dict[f"{col}_dist"] = dist


def optionally_top_n_peaks(num_of_peaks, peak_indices, peaks_meta):
    if num_of_peaks > 0 and len(peak_indices) > 0:
        proms = peaks_meta["prominences"]

        top_peaks = numpy.argsort(-numpy.array(proms))[:num_of_peaks]

        top_peaks = sorted(top_peaks)

        peak_indices = [peak_indices[i] for i in top_peaks]

        for col in "prominences", "right_bases", "left_bases", "widths", "width_heights", "left_ips", "right_ips", "plateau_sizes", "left_edges", "right_edges":
            meta_col = peaks_meta[col]
            peaks_meta[col] = [meta_col[i] for i in top_peaks]

    return peak_indices


def create_peak_text_and_metadata(peak_indices, peaks_meta, sentence_nums, sentence_text, story_id, type, field):
    hover_text = []
    peaks_list = []

    for i, ind in enumerate(peak_indices):

        peak_dict = {}

        peak_dict["story_id"] = story_id
        peak_dict["field"] = field
        peak_dict["text"] = []
        peak_dict["sentence_nums"] = []

        left_base = peaks_meta["left_bases"][i]
        right_base = peaks_meta["right_bases"][i]
        text = ""

        for j in range(left_base, right_base):
            j = min(max(j, 0), len(sentence_nums) - 1)
            wrapper = TextWrapper(initial_indent="<br>", width=80)

            if j == ind:
                peak_dict["sentence"] = sentence_text[j]
                peak_dict["sentence_num"] = sentence_nums[j]

                wrapper = TextWrapper(initial_indent="<br>")
                wrapped_text = wrapper.fill(f"<b>{sentence_nums[j]} - {sentence_text[j]}</b>")

                text += wrapped_text
            else:

                wrapped_text = wrapper.fill(f"{sentence_nums[j]} - {sentence_text[j]}")

                text += wrapped_text

            peak_dict["text"].append(sentence_text[j])
            peak_dict["sentence_nums"].append(sentence_nums[j])

        prominance = peaks_meta["prominences"][i]
        width = peaks_meta["widths"][i]
        importance = prominance * width

        peak_dict["prominence"] = prominance
        peak_dict["width"] = width
        peak_dict["importance"] = importance
        peak_dict["type"] = type

        text += "<br>"
        text += f"<br>Prominence: {prominance} <br>Width: {width} <br>Importance: {importance}"

        peaks_list.append(peak_dict)
        hover_text.append(text)
    return hover_text, peaks_list

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


def create_analysis_output(args):
    print(args)

    ensure_dir(args["output_dir"])


    df_list = [r for r in extract_rows(args)]

    position_df = pandas.concat(df_list)
    position_df.fillna(0.0)
    print(position_df)

    turning_points_df = None
    if "turning_points_csv" in args and args["turning_points_csv"] is not None:
        turning_points_df = pd.read_csv(args["turning_points_csv"])
        turning_points_df = turning_points_df.fillna(value=0.0)
        print(turning_points_df)

    metric_columns = [i for i in list(position_df.columns) if i.startswith("metric")]
    metric_columns = sorted(metric_columns)
    print(metric_columns)

    
    create_story_plots(position_df=position_df, turning_points_df=turning_points_df, args=args)


create_analysis_output(vars(args))
