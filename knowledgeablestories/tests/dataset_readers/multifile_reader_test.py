import os
import pathlib
from typing import cast

import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data import Vocabulary

from knowledgeablestories.dataset_readers.book_summaries_reader import CmuBookHierarchyReader, CmuBookLMReader
from knowledgeablestories.dataset_readers.cbt_reader import CbtHierarchyReader, CbtLMReader
from knowledgeablestories.dataset_readers.movie_summaries_reader import CmuMovieLMReader, CmuMovieHierarchyReader
from knowledgeablestories.dataset_readers.multifile_reader import MultifileHierarchyReader, MultifileLMReader

AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()

class TestMultifileDatasetReader(AllenNlpTestCase):

    def test_hierarchy_schmoop(self):
        reader = MultifileHierarchyReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/schmoop_xmas_carol_0.txt.utf8"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 12

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)


    def test_lm_schmoop(self):
        reader = MultifileLMReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/schmoop_xmas_carol_0.txt.utf8"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 12

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)

    def test_hierarchy_books_corpus(self):
        reader = MultifileHierarchyReader()
        instances = reader.read(
            str(
                AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/575718__alice-in-demonland.txt"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 47

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)

    def test_lm_books_corpus(self):
        reader = MultifileLMReader()
        instances = reader.read(
            str(
                AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/575718__alice-in-demonland.txt"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 47

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)
            instance.index_fields(Vocabulary())
            instance_tensor_dict = instance.as_tensor_dict()
            print(instance_tensor_dict)