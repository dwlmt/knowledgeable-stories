import pathlib

from allennlp.common.testing import ModelTestCase, AllenNlpTestCase

from knowledgeablestories.dataset_readers.atomic_reader import AtomicDatasetReader
from knowledgeablestories.models import KnowledgeableStoriesModel

AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()

class TestMaskedLanguageModel(ModelTestCase):
    def setUp(self):
        super().setUp()
        self.set_up_model(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/config/knowledgeable_story_model_test.jsonnet",
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/atomic_small.csv",
        )

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)