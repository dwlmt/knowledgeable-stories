import os
import pathlib
from typing import cast

import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase, test_case
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance

from knowledgeablestories.dataset_readers.roc_reader import RocLMReader, RocHierarchyReader
from knowledgeablestories.dataset_readers.writing_prompts_reader import WritingPromptsLMReader

AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()

class TestWritingPromptsLMDatasetReader(AllenNlpTestCase):

    def test_hierarchy(self):
        reader = WritingPromptsLMReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/writing_prompts_25"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 26

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)