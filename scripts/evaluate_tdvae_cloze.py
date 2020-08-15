import csv

import fire
from jsonlines import jsonlines
from tqdm import tqdm

class EvalTdvaeCloze(object):
    """Evaluate TDVAE Cloze
    """
    def eval(self, prediction_json: str, output_file: str, accuracy_field: str = "whole_story_smaller"):
        """ Evaluate TDVAE

            @param prediction_json: Source with the prediction json data.
            @param output_file
        """
        print(f"Evaluate TDVAE Cloze from {prediction_json}, to {output_file}")

        with jsonlines.open(prediction_json) as reader:

            total_rows = 0
            sum_dict = {}

            ranked_results_counts = {}
            for i, obj in tqdm(enumerate(reader)):

                if i == 0:
                    sum_dict = obj[accuracy_field]
                else:
                    accuracy = obj[accuracy_field]
                    sum_dict = {k: sum_dict.get(k, 0) + accuracy.get(k, 0) for k in set(sum_dict) | set(accuracy)}

                for rank_key, rank_val in obj["ranked_results"].items():

                    if rank_key not in ranked_results_counts:
                        ranked_results_counts[rank_key] = 0

                    ranked_results_counts[rank_key] += int(rank_val)

                total_rows += 1

        accuracy_dict = {}
        for k, v in sum_dict.items():
            accuracy_dict[k] = float(v) / float(total_rows)

        for k, v in ranked_results_counts.items():
            ranked_results_counts[k] = float(v) / float(total_rows)

        with open(output_file, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["metric", accuracy_field])

            for k, v in accuracy_dict.items():
                csv_writer.writerow([k, v])

            for k, v in ranked_results_counts.items():
                csv_writer.writerow([k, v])


if __name__ == '__main__':
    fire.Fire(EvalTdvaeCloze)
