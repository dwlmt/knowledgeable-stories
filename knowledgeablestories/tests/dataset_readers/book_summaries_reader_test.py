import os
import pathlib
from typing import cast

import pytest

from allennlp.common.util import ensure_list
from allennlp.common.testing import AllenNlpTestCase

from knowledgeablestories.dataset_readers.book_summaries_reader import CmuBookHierarchyReader, CmuBookLMReader
from knowledgeablestories.dataset_readers.movie_summaries_reader import CmuMovieLMReader, CmuMovieHierarchyReader


AllenNlpTestCase.MODULE_ROOT = (pathlib.Path(__file__).parent / ".." / ".." / "..").resolve()

class TestMovieSummariesDatasetReader(AllenNlpTestCase):

    def test_hierarchy(self):
        reader = CmuBookHierarchyReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/booksummaries_20.txt"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 27

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)


    def test_lm(self):
        reader = CmuBookLMReader()
        instances = reader.read(
            str(AllenNlpTestCase.MODULE_ROOT) + "/knowledgeablestories/tests/fixtures/data/booksummaries_20.txt"
        )
        instances = ensure_list(instances)

        print(instances)

        assert len(instances) == 27

        for instance in instances:
            print(instance)
            print(instance["metadata"].metadata)