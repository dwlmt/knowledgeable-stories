import os
import pathlib
from typing import cast

import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase, test_case
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance

from knowledgeablestories.dataset_readers.roc_reader import RocLMReader

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

    def test_read_test(self):
        reader = RocLMReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/roc_val_50.csv"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 50

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
