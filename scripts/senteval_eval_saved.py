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
import torch

def add_bool_arg(parser, name, default=False):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=name.replace('-','_'), action='store_true')
    group.add_argument('--no-' + name, dest=name.replace('-','_'), action='store_false')
    parser.set_defaults(**{name:default})


def main():
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings',
                        help='Path to JSON vectors file.')
    parser.add_argument('--output-file',
                        help='Write output to file.')
    parser.add_argument('--dim-size', default=3072, type=int,
                        help='size of the default dimension size.')
    parser.add_argument('--embedding-name', default="sentences_encoded", type=str,
                        help='Name for the embedding attribute.')
    parser.add_argument('-t', '--tasks',
                        help='A comma-separated list of tasks.')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Do not use GPU (turn off PyTorch).')

    add_bool_arg(parser, 'cat-minus-vector')

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

        if not len(sent2emb) > 0:

            with jsonlines.open(args.embeddings) as reader:
                for batch in reader:
                    if "sentences" in batch:
                        sentences = batch["sentences"]

                        for sentence in sentences:
                            with torch.no_grad():
                                print(f"Load sentence embedding for: {sentence['text']}")
                                sent_list = sentence[args.embedding_name]
                                sent_one = torch.tensor(sent_list[: 1024])
                                sent_two = torch.tensor(sent_list[1024 :])

                                if args.cat_minus_vector:
                                    sent_emb = torch.cat((sent_one, sent_two, abs(sent_one - sent_two)))
                                else:
                                    sent_emb = torch.cat((sent_one, sent_two))

                                sent2emb[sentence["text"]] = sent_emb.cpu().numpy()

        else:
            print(f"Loaded sentences in dict: {len(sent2emb)}")

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

    params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 10}

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    if args.tasks is not None:
        transfer_tasks = args.tasks.split(',')
    else:
        transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',  #'SNLI',
                          'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', #'ImageCaptionRetrieval',
                          'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                          'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                          'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    results = se.eval(transfer_tasks)
    with open(args.output_file, 'w') as outfile:
        json.dump(results, outfile , cls=NumpyEncoder)

    sys.stdout.write('\n')

if __name__ == '__main__':
    main()
