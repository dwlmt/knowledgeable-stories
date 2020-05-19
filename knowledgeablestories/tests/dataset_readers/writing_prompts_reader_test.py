import pathlib

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary

import os

from knowledgeablestories.dataset_readers.writing_prompts_reader import WritingPromptsLMReader, \
    WritingPromptsHierarchyReader


AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()


class TestWritingPromptsLMDatasetReader(AllenNlpTestCase):

    def test_hierarchy(self):
        reader = WritingPromptsHierarchyReader()
        wp_path = f'{str(AllenNlpTestCase.MODULE_ROOT)}\\{str(os.path.join("knowledgeablestories", "tests", "fixtures", "data", "writing_prompts_25"))}'
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + wp_path
        )
        instances = ensure_list(instances)

        print(instances)

        #assert len(instances) == 40

        for instance in instances:
            print(instance)
            #print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)

    def test_lm(self):
        reader = WritingPromptsLMReader()
        wp_path = f'{str(AllenNlpTestCase.MODULE_ROOT)}\\{str(os.path.join("knowledgeablestories", "tests", "fixtures", "data", "writing_prompts_25"))}'
        instances = reader.read(
            wp_path
        )
        instances = ensure_list(instances)

        print(instances)

        #assert len(instances) == 246

        for instance in instances:
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)
            print(instance)
            print(instance["metadata"].metadata)
