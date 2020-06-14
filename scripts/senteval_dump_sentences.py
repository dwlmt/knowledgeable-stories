"""
A script to dump all sentences (tokenized) to standard output.
"""

from __future__ import absolute_import, division, unicode_literals

import argparse
import logging
import os
import sys

# Set PATHs
PATH_TO_SENTEVAL = os.path.dirname(os.getenv("PATH_TO_SENTEVAL", default="/home/s1569885/git/SentEval/"))
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL, 'data')

sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


def main():
    logging.basicConfig(format='%(asctime)s : %(message)s',
                        level=logging.DEBUG)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-t", "--tasks",
                        help="a comma-separated list of tasks")
    args = parser.parse_args()

    def prepare(params, samples):
        for sent in samples:
            if sys.version_info < (3, 0):
                sent = [w.decode('utf-8') if isinstance(w, str) else w for w in sent]
                print(' '.join(sent).encode('utf-8'))
            else:
                sent = [w.decode('utf-8') if isinstance(w, bytes) else w for w in sent]
                print(' '.join(sent))

    def batcher(params, batch):
        # Block evaluation and continue with the next task.
        raise Done

    params_senteval = {
        'task_path': PATH_TO_DATA
    }

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    if args.tasks is not None:
        transfer_tasks = args.tasks.split(',')
    else:
        transfer_tasks = ['CR', 'MR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                          'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'ImageCaptionRetrieval',
                          'STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                          'Length', 'WordContent', 'Depth', 'TopConstituents', 'BigramShift', 'Tense',
                          'SubjNumber', 'ObjNumber', 'OddManOut', 'CoordinationInversion']

    for task in transfer_tasks:
        try:
            se.eval([task])
            raise RuntimeError(task + " not completed")
        except Done:
            pass


class Done(Exception):
    pass


if __name__ == "__main__":
    main()
