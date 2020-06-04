import csv

import fire
from jsonlines import jsonlines
from tqdm import tqdm


class ConvertSentEval(object):
    def convert(self, source: str, target: str):

        print(f"Convert from text to json: {source}, to {target}")

        with open(source) as reader:
            with jsonlines.open(target, mode='w') as writer:
                for line in reader:
                    writer.write({"text": line.strip('\n')})

if __name__ == '__main__':
    fire.Fire(ConvertSentEval)
