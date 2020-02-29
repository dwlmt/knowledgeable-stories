import pathlib

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary

from knowledgeablestories.dataset_readers.writing_prompts_reader import WritingPromptsLMReader, \
    WritingPromptsHierarchyReader
from knowledgeablestories.dataset_readers.writing_prompts_reader_new import WritingPromptsLMReaderNew, \
    WritingPromptsHierarchyReaderNew

AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()


class TestWritingPromptsLMDatasetReader(AllenNlpTestCase):

    def test_hierarchy(self):
        reader = WritingPromptsHierarchyReaderNew()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/writing_prompts_25"
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
        reader = WritingPromptsLMReaderNew()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/writing_prompts_25"
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
