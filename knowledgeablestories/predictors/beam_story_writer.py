''' Simple Greedy story generation
'''
import copy
import os
from operator import itemgetter

from allennlp.common import JsonDict
from allennlp.data import DatasetReader
from allennlp.models import Model
from allennlp.predictors import Predictor

from knowledgeablestories import KnowledgeablePredictor


@Predictor.register('know_beam_story_writer')
class KnowledgeableBeamStoryWriterPredictor(KnowledgeablePredictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model=model, dataset_reader=dataset_reader)

        self._keep_top_n = int(os.getenv("STORY_WRITER_KEEP_TOP_N", default=1))

    def predict_json(self, inputs: JsonDict) -> JsonDict:

        self._split_sentences_if_required(inputs)

        # Only apply sentence generation to the last one.
        inputs["rollout_indices"] = [len(inputs["sentences"]) - 1]

        copied_input_sentences = copy.deepcopy(inputs["sentences"])
        copied_input = copy.deepcopy(inputs)

        if "levels_to_rollout" in inputs:
            self._num_levels_rollout = inputs["levels_to_rollout"]

        rollout_outputs = self.rollout_prediction(inputs)

        last_sentence = rollout_outputs["sentences"][-1]
        print("Last Sentence", last_sentence)

        current_hypothesis = []
        hypotheses = []

        def tree_search(current_sentence, current_hypothesis):
            current_hypothesis.append(current_sentence)

            child_sentences = []
            if "sentences" in current_sentence:
                child_sentences = current_sentence["sentences"]
                del current_sentence["sentences"]
            else:
                hypotheses.append(copy.deepcopy(current_hypothesis))
                print([s["text"] for s in current_hypothesis])

            for sent in child_sentences:
                tree_search(sent, current_hypothesis)

            current_hypothesis.pop(-1)

        tree_search(last_sentence, current_hypothesis)

        # Filter shorter hypotheses.
        hypotheses = [h[1:len(h)] for h in hypotheses if len(h) > self._num_levels_rollout]

        full_stories = []
        for i, hyp in enumerate(hypotheses):

            con_hyp = []
            for i, h in enumerate(hyp, start=len(copied_input_sentences)):
                processed_sentence = {"sentence_num": i, "text": h["text"],
                                      "chain_log_prob": h["parent_relation_metrics"]["chain_log_prob"],
                                      "chain_prob": h["parent_relation_metrics"]["chain_prob"],
                                      "log_prob": h["parent_relation_metrics"]["log_prob"],
                                      "prob": h["parent_relation_metrics"]["prob"]}
                con_hyp.append(processed_sentence)

            last_sentence = con_hyp[-1]

            joined_text = copied_input_sentences + con_hyp

            story = {"index": i, "sentences": joined_text, "prob": last_sentence["prob"],
                     "log_prob": last_sentence["log_prob"]}

            full_stories.append(story)

        # Use log prob to sort the full story output.
        full_stories = sorted(full_stories, key=itemgetter("log_prob"), reverse=True)

        if len(full_stories) > self._keep_top_n:
            full_stories = full_stories[0: self._keep_top_n]

        copied_input["generated"] = full_stories

        return copied_input
