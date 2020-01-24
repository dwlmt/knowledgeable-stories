from typing import cast

import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance

from knowledgeablestories.readers.atomic_reader import AtomicDatasetReader

class TestAtomicDatasetReader:
    def test_read(self, lazy):
        reader = AtomicDatasetReader()
        instances = reader.read(
            str(AllenNlpTestCase.FIXTURES_ROOT / "data" / "atomic_small.csv")
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 100