import pathlib

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list

from knowledgeablestories.dataset_readers.movie_summaries_reader import CmuMovieLMReader, CmuMovieHierarchyReader
from knowledgeablestories.dataset_readers.writing_prompts_reader import WritingPromptsLMReader, \
    WritingPromptsHierarchyReader

AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()


class TestWritingPromptsLMDatasetReader(AllenNlpTestCase):

    def test_hierarchy(self):
        reader = WritingPromptsHierarchyReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/writing_prompts_25"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 41

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)

    def test_lm(self):
        reader = WritingPromptsLMReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/writing_prompts_25"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 41

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
