import pathlib

from allennlp.common.testing import AllenNlpTestCase
from allennlp.common.util import ensure_list
from allennlp.data import Vocabulary

from knowledgeablestories.dataset_readers.roc_reader import RocLMReader, RocHierarchyReader

AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()


class TestRocLMDatasetReader(AllenNlpTestCase):
    def test_read_train(self):
        reader = RocLMReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/roc_train_50.csv"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 50

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)

    def test_read_test(self):
        reader = RocLMReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/roc_val_50.csv"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 50

        for instance in instances:
            print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)

    def test_read_train_hierarchy(self):
        reader = RocHierarchyReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/roc_train_50.csv"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 50

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)

    def test_read_test_hierarchy(self):
        reader = RocHierarchyReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/roc_val_50.csv"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 50

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            print(instance.get_padding_lengths())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)
