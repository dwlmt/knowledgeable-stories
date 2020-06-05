import csv

import fire
import more_itertools
from jsonlines import jsonlines
from tqdm import tqdm


class ConvertSentEval(object):
    def convert(self, source: str, target: str, batch_size: int = 100):

        print(f"Convert from text to json: {source}, to {target}")

        with open(source) as reader:
            all_lines = list(reader.readlines())
            with jsonlines.open(target, mode='w') as writer:

                for batch_id, batch in enumerate(more_itertools.chunked(all_lines, batch_size)):

                    story_dict = {}
                    story_dict["story_id"] = batch_id

                    sentences = []

                    for j, sentence in enumerate(sentences):
                        sentences.append({"sentence_num": j,"text": sentence.strip('\n')})

                    story_dict["sentences"] = sentences

                    writer.write(story_dict)

if __name__ == '__main__':
    fire.Fire(ConvertSentEval)
