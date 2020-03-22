import argparse
import multiprocessing
import os
from pathlib import Path

import dask
import dask.array
import dask.bag
import hdbscan
import numpy
from joblib import dump
from jsonlines import jsonlines
from sklearn.preprocessing import normalize
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Extract JSON vectors and perform dimensionality reduction.')
parser.add_argument('--source-json', required=True, type=str, help="JSON file to process.")
parser.add_argument('--output-file', required=True, type=str, help="Output, saved as a Parquet file")
parser.add_argument('--output-cluster-path', required=True, type=str, help="Pickle the clusters to reuse.")
parser.add_argument('--umap-n-neighbours', default=100, type=int, help="The number of neighbours.")
parser.add_argument('--umap-min-dist', default=0.0, type=float, help="Controls how clumpy umap will cluster points.")
parser.add_argument('--dissimilarity-metric', default=["cosine", "euclidean"], nargs="+", type=str,
                    help="The dissimilarity or distance metric to use.")
parser.add_argument('--dim-reduction-components', default=[2], type=int, nargs="+",
                    help="The number of components to reduce to.")
parser.add_argument('--metadata-columns', default=["sentence_num", "text", "tokens"], nargs="+", type=str,
                    help="The vector columns to process.")
parser.add_argument('--vector-columns', default=["passage_autoencoded_mu", "passage_autoencoded_var",
                                                 "passage_autoencoded_diff_mu", "passage_autoencoded_diff_var",
                                                 "sentence_autoencoded_mu", "sentence_autoencoded_var"], nargs="+",
                    type=str,
                    help="The vector columns to process.")
parser.add_argument('--cluster-columns', default=["passage_autoencoded_mu",
                                                  "passage_autoencoded_diff_mu", "sentence_autoencoded_mu"], nargs="+",
                    type=str,
                    help="The vector columns to process.")
parser.add_argument('--min-cluster-size', default=20, type=int, help="Min size fo each cluster.")
parser.add_argument('--max_num-stories', default=100, type=int, help="Max number of stories to process")

args = parser.parse_args()


def cluster_vectors(args):
    Path(args["output_cluster_path"]).mkdir(parents=True, exist_ok=True)
    Path(os.path.dirname(args["output_file"])).mkdir(parents=True, exist_ok=True)

    # Extract and process the raw data and convert to a dask
    original_df = dask.bag.from_sequence(
        extract_rows(args))
    original_df = original_df.to_dataframe()

    for field in args["cluster_columns"]:

        for sim_metric in args["dissimilarity_metric"]:

            print(f"HDBSCAN Clustering for : {field}, {sim_metric}")

            metric = sim_metric

            vector_data = dask.array.from_array(numpy.array(original_df[field].compute().values.tolist()),
                                                chunks=(1000, 1000))

            if sim_metric == "cosine":
                vector_data = normalize(vector_data, norm='l2')
                metric = "euclidean"

            clusterer = hdbscan.HDBSCAN(algorithm='best', metric=metric,
                                        min_cluster_size=args["min_cluster_size"],
                                        core_dist_n_jobs=multiprocessing.cpu_count() - 1)

            clusterer.fit(vector_data)

            dump(clusterer, f"{args['output_cluster_path']}/hdbscan_{field}_{sim_metric}.joblib")

            labels = clusterer.labels_,
            probabilities = clusterer.probabilities_,
            persistence = clusterer.cluster_persistence_,
            condenses_tree = clusterer._condensed_tree,
            single_linkage_tree = clusterer._single_linkage_tree,
            min_spanning_tree = clusterer._min_spanning_tree

            print("Cluster Output", labels, probabilities, persistence, condenses_tree, single_linkage_tree,
                  min_spanning_tree)


def extract_rows(args):
    index_counter = 0
    with jsonlines.open(args['source_json']) as reader:
        for i, obj in tqdm(enumerate(reader)):
            for child in obj["sentences"]:
                yield {**{k: child[k] for k in args["metadata_columns"]},
                       **{k: numpy.array(child[k], dtype=numpy.float32) for k in args["vector_columns"]}}

                index_counter += 1

            if i == args["max_num_stories"]:
                break


cluster_vectors(vars(args))
