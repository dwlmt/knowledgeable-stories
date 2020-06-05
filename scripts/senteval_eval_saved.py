from __future__ import absolute_import, division, unicode_literals

import argparse
import json
import logging
import os
import sys

import numpy as np

# Set PATHs
from jsonlines import jsonlines

PATH_TO_SENTEVAL = os.path.dirname(os.getenv("PATH_TO_SENTEVAL", default="/home/s1569885/git/SentEval/"))
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, 'data')

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def main():
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings',
                        help='Path to JSON vectors file.')
    parser.add_argument('--dim-size', default=2048, type=int,
                        help='size of the default dimension size.')
    parser.add_argument('--embedding-name', default="sentences_encoded", type=str,
                        help='Name for the embedding attribute.')
    parser.add_argument('-t', '--tasks',
                        help='A comma-separated list of tasks.')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Do not use GPU (turn off PyTorch).')
    args = parser.parse_args()

    sent2emb = {}

    def join_sentence(sent):
        if sys.version_info < (3, 0):
            sent = [w.decode('utf-8') if isinstance(w, str) else w for w in sent]
        else:
            sent = [w.decode('utf-8') if isinstance(w, bytes) else w for w in sent]
        return ' '.join(sent)

    def prepare(params, samples):
        # Build the mapping from sentences to embeddings
        sent2emb.clear()

        with jsonlines.open(args.embeddings) as reader:
            for batch in reader:
                if "sentences" in batch:
                    sentences = batch["sentences"]

                    for sentence in sentences:
                        print(f"Load sentence embedding for {sentence['text']}")
                        sent2emb[sentence["text"]] = np.array(sentence[args.embedding_name])

    def batcher(params, batch):
        embeddings_list = []

        for sent in batch:
            joined_sent = join_sentence(sent)

            if joined_sent in sent2emb:
                embeddings_list.append(sent2emb[joined_sent])
            else:
                print(f"Random tensor for: {joined_sent}")
                embeddings_list.append(np.random.rand(args.dim_size))

        embeddings = np.stack(embeddings_list)
        if len(embeddings.shape) != 2:
            embeddings = embeddings.reshape(len(embeddings), -1)

        return embeddings


    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                            'tenacity': 3, 'epoch_size': 2}

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    if args.tasks is not None:
        transfer_tasks = args.tasks.split(',')
    else:
        transfer_tasks = se.list_tasks

    results = se.eval(transfer_tasks)
    json.dump(results, sys.stdout, skipkeys=True)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()