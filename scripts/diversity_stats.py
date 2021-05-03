from collections import Counter
from pathlib import Path
from random import shuffle
from typing import List

import fire
import jsonlines as jsonlines
import pandas
import sklearn
from sklearn.metrics.pairwise import pairwise_distances


class DiversityStats(object):
    ''' Evaluate salience scores.
    '''

    def evaluate(self, src_json: str, output_dir: str):

        Path(f"{output_dir}/").mkdir(parents=True, exist_ok=True)

        import spacy

        nlp = spacy.load("en_core_web_sm")

        import neuralcoref
        neuralcoref.add_to_pipe(nlp)

        pos_counters = {}
        coreference_list = []
        total_tokens = 0
        story_lengthes = []

        num_stories = 0
        with jsonlines.open(f"{src_json}") as reader:

            for obj in reader:

                num_stories += 1

                print(f"OBJ: {obj}")

                text = None
              
                if "generated" in obj:
                    if len(obj["generated"]) > 0:
                        generated = obj["generated"]
                        generated = generated[0]
                        if isinstance(generated, dict):
                            sentences = generated["sentences"]
                        else:
                            sentences = generated
                        print(generated)
                        text = " ".join([s["text"] for s in sentences])
                elif "sentences" in obj:
                    if len(obj["sentences"]) > 0:
                        text = " ".join([s["text"] for s in obj["sentences"]])
                elif "passage" in obj:
                    text = obj["passage"]

                if text != None:

                    text = text.replace("<|endofsentence|>", " ")
                    print(f"TEXT: {text}")

                    doc = nlp(text)
                    num_tokens = len(doc)
                    total_tokens += num_tokens
                    story_lengthes.append(num_tokens)

                    if len(doc._.coref_clusters) > 0:
                        print(f"Coreference docs: {doc._.coref_clusters}")
                        number_of_coreferences = len(doc._.coref_clusters)
                        mention_chain_list = []
                        for cluster in doc._.coref_clusters:
                            mention_chain_list.append(float(len(cluster)))
                            for mention in cluster:
                                print(f"Mentions {mention}")

                        mention_chain_avg = sum(mention_chain_list) / float(len(mention_chain_list))

                        coreference_list.append({"num_coreferences": number_of_coreferences,"mention_chain_len": mention_chain_avg})

                    for token in doc:
                        #print(token.text, token.pos_, token.dep_)

                        if token.pos_ not in pos_counters:
                            pos_counters[token.pos_] = Counter()

                        pos_counters[token.pos_][token.lemma_] += 1

        #print(pos_counters)

        with jsonlines.open(f'{output_dir}/pos_counters.jsonl', mode='w') as writer:
            for k, v in pos_counters.items():
                writer.write({"pos": k, "counts": v })

        diversity_list = []
        for k, v in pos_counters.items():
            num_of_tokens = float(len(v))
            diversity_avg = num_of_tokens / total_tokens
            diversity_per_story_avg = num_of_tokens / num_stories
            diversity_list.append({"pos": k, "avg_per_token": diversity_avg, "avg_per_story": diversity_per_story_avg, "unique_tokens": num_of_tokens})

        diversity_df = pandas.DataFrame(diversity_list)
        diversity_df.to_csv(f'{output_dir}/diversity.csv')

        coreference_df = pandas.DataFrame(coreference_list)
        coreference_stats_df = coreference_df.describe()
        coreference_stats_df.to_csv(f'{output_dir}/coreference.csv')




if __name__ == '__main__':
    fire.Fire(DiversityStats)