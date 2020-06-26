import os
import pathlib
from typing import cast

import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase, test_case
from allennlp.data import Vocabulary

from knowledgeablestories.dataset_readers.swag_reader import SwagKnowDatasetReader

AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()

class TestSwagDatasetReader(AllenNlpTestCase):
    def test_read(self):
        reader = SwagKnowDatasetReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/swag_train_50.csv"
        )
        instances = ensure_list(instances)

        assert len(instances) == 50 # Each relation is expanded out.

        for instance in instances:

            print(instance)
            print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)
