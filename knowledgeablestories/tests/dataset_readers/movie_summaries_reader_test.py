import os
import pathlib
from typing import cast

import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary

from knowledgeablestories.dataset_readers.movie_summaries_reader import CmuMovieLMReader, CmuMovieHierarchyReader


AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()

class TestMovieSummariesDatasetReader(AllenNlpTestCase):

    def test_hierarchy(self):
        reader = CmuMovieHierarchyReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/plot_summaries_20"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 21

        for instance in instances:
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)
            print(instance)
            print(instance["metadata"].metadata)


    def test_lm(self):
        reader = CmuMovieLMReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/plot_summaries_20"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 21

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)