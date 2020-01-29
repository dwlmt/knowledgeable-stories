import os
import pathlib
from typing import cast

import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase, test_case
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance

from knowledgeablestories.dataset_readers.atomic_reader import AtomicDatasetReader

AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()

class TestAtomicDatasetReader(AllenNlpTestCase):
    def test_read(self):
        reader = AtomicDatasetReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/atomic_small.csv"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 208 # Each relation is expanded out.

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
