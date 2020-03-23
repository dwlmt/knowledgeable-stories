import argparse
import multiprocessing
from pathlib import Path

import dask
import dask.array
import dask.bag
import hdbscan
import numpy
import umap
from joblib import dump
from jsonlines import jsonlines
from sklearn import cluster
from sklearn.preprocessing import normalize
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='Extract JSON vectors and perform dimensionality reduction.')
parser.add_argument('--source-json', required=True, type=str, help="JSON file to process.")
parser.add_argument('--output-cluster-path', required=True, type=str, help="Pickle the clusters to reuse.")
parser.add_argument('--umap-n-neighbours', default=100, type=int, help="The number of neighbours.")
parser.add_argument('--umap-min-dist', default=0.0, type=float, help="Controls how clumpy umap will cluster points.")
parser.add_argument('--kmeans-ncentroids', default=[32], type=int, nargs="+", help="Number of K-means centroids.")
parser.add_argument('--kmeans-init', default=32, type=int, help="Number of times to run K-Means.")
parser.add_argument('--kmeans-iterations', default=300, type=int, help="Max number of K-means iterations.")
parser.add_argument('--dim-reduction-component', default=2, type=int, help="The number of components to reduce to.")
parser.add_argument('--dissimilarity-metric', default=["cosine", "euclidean"], nargs="+", type=str,
                    help="The dissimilarity or distance metric to use.")
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
parser.add_argument('--max-num-stories', default=100, type=int, help="Max number of stories to process")
parser.add_argument("--dont-save-csv", default=False, action="store_true", help="Don't save summary fields to csv.")

args = parser.parse_args()


def cluster_vectors(args):
    Path(args["output_cluster_path"]).mkdir(parents=True, exist_ok=True)

    # Extract and process the raw data and convert to a dask
    original_df = dask.bag.from_sequence(
        extract_rows(args))
    original_df = original_df.to_dataframe()

    export_fields = ["story_id", "sentence_num", "text"]
    export_df = original_df[export_fields].compute()

    for col in args["cluster_columns"]:

        for sim_metric in args["dissimilarity_metric"]:

            print(f"HDBSCAN Clustering for : {col}, {sim_metric}")

            metric = sim_metric

            vector_data = dask.array.from_array(numpy.array(original_df[col].compute().values.tolist()),
                                                chunks=(1000, 1000))

            if sim_metric == "cosine":
                vector_data = normalize(vector_data, norm='l2')
                metric = "euclidean"

            hdbscan_clusterer = hdbscan.HDBSCAN(algorithm='best', metric=metric,
                                                min_cluster_size=args["min_cluster_size"],
                                                core_dist_n_jobs=multiprocessing.cpu_count() - 1)

            hdbscan_clusterer.fit(vector_data)

            dump(hdbscan_clusterer, f"{args['output_cluster_path']}/hdbscan_{col}_{sim_metric}.joblib")

            label_col = f"hdbscan_{col}_{sim_metric}_label"
            export_df[label_col] = hdbscan_clusterer.labels_.tolist()
            prob_col = f"hdbscan_{col}_{sim_metric}_probability"
            export_df[prob_col] = hdbscan_clusterer.probabilities_.tolist()

            dim = args["dim_reduction_component"]

            reduced_vector_data = umap.UMAP(n_neighbors=args["umap_n_neighbours"], min_dist=args["umap_min_dist"],
                                            n_components=dim, metric=sim_metric).fit_transform(
                vector_data)
            x, y = zip(*reduced_vector_data.tolist())
            x_col = f"{col}_{x}"
            y_col = f"{col}_{y}"
            export_df[x_col] = x
            export_df[y_col] = y

        # Kmeans clusters.
        for dim in args["kmeans_ncentroids"]:
            kmeans_clusterer = cluster.KMeans(n_clusters=dim, n_init=args["kmeans_init"],
                                              max_iter=args["kmeans_iterations"], n_jobs=-1)
            labels = kmeans_clusterer.fit_predict(vector_data)
            dump(kmeans_clusterer, f"{args['output_cluster_path']}/hdbscan_{col}_{dim}.joblib")

            label_col = f"kmeans_{col}_{dim}_label"
            export_df[label_col] = labels.tolist()

    if not args["dont_save_csv"]:
        export_df.to_csv(f"{args['output_cluster_path']}/cluster_export.xz")

def extract_rows(args):
    index_counter = 0
    with jsonlines.open(args['source_json']) as reader:
        for i, obj in tqdm(enumerate(reader)):
            story_id = obj["story_id"]
            for child in obj["sentences"]:
                processed_dict = {**{k: child[k] for k in args["metadata_columns"]},
                                  **{k: numpy.array(child[k], dtype=numpy.float32) for k in args["vector_columns"]}}
                processed_dict["story_id"] = story_id
                yield processed_dict

                index_counter += 1

            if i == args["max_num_stories"]:
                break

cluster_vectors(vars(args))
