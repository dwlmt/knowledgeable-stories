import collections
import csv
from typing import List, OrderedDict

import fire
import more_itertools
from jsonlines import jsonlines
from tqdm import tqdm


def cleanup_text(param):
    if param is None or len(param) == 0:
        return param
    for r in ["\n","<|endofsentence|>","<|endoftext|>","<newline>"]:
        param = param.replace(r, "")
    return param


class StoryEvaluationTasks(object):
    def create(self, prompts_json: str, gold_json: str, models_json: List[str], models_types: List[str],
               output_file: str):

        if isinstance(models_json, str):
            models_json = [models_json]

        if isinstance(models_types, str):
            models_types = [models_types]

        assert len(models_json) == len(models_types), "Models and types provided must be the same length."

        number_of_fields = len(models_json) + 1

        prompt_dict = collections.OrderedDict()
        gold_dict = collections.OrderedDict()

        models_dict = collections.OrderedDict()

        with jsonlines.open(prompts_json) as reader:
            for obj in reader:
                prompt_dict[obj["story_id"]] = {"story_id": obj["story_id"], "passage": cleanup_text(obj["passage"])}

        with jsonlines.open(gold_json) as reader:
            for obj in reader:
                gold_dict[obj["story_id"]] =  {"story_id": obj["story_id"], "passage": cleanup_text(obj["passage"])}

        models_dict["gold"] = gold_dict
        for m, t in zip(models_json, models_types):

            m_dict = collections.OrderedDict()
            with jsonlines.open(m) as reader:
                for obj in reader:
                    m_dict[obj["story_id"]] = {"story_id": obj["story_id"]}

                    sentences = []
                    for s in obj["generated"][0]["sentences"]:
                        sentences.append(cleanup_text(s["text"]))

                    m_dict[obj["story_id"]]["passage"] = " ".join(sentences)

            models_dict[t] = m_dict

        print(f"Prompts: {prompt_dict.values()}")
        print(f"Gold: {gold_dict.values()}")
        print(f"Models: {models_dict.values()}")

if __name__ == '__main__':
    fire.Fire(StoryEvaluationTasks)