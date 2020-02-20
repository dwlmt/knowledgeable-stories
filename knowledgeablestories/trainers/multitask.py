"""

This is an  from https://github.com/allenai/kb/blob/2e1b1c7fee48d007fbd716dbf0b58c50d4ca01a5/kb/multitask.py

Used in the KnowBert paper:

@inproceedings{Peters2019KnowledgeEC,
  author={Matthew E. Peters and Mark Neumann and Robert L Logan and Roy Schwartz and Vidur Joshi and Sameer Singh and Noah A. Smith},
  title={Knowledge Enhanced Contextual Word Representations},
  booktitle={EMNLP},
  year={2019}
}

"""

from typing import Dict, List, Iterable

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import Instance
from allennlp.data.iterators import DataIterator
from allennlp.data import Vocabulary

import numpy as np

import torch


class MultitaskDataset:
    def __init__(self, datasets: Dict[str, Iterable[Instance]],
                       datasets_for_vocab_creation: List[str]):
        self.datasets = datasets
        self.datasets_for_vocab_creation = datasets_for_vocab_creation

    def __iter__(self):
        # with our iterator, this is only called for vocab creation
        for key in self.datasets_for_vocab_creation:
            for instance in self.datasets[key]:
                yield instance


@DatasetReader.register("multitask_reader")
class MultitaskDatasetReader(DatasetReader):
    def __init__(self,
                 dataset_readers: Dict[str, DatasetReader],
                 datasets_for_vocab_creation: List[str]) -> None:
        super().__init__(False)
        self.dataset_readers = dataset_readers
        self.datasets_for_vocab_creation = datasets_for_vocab_creation

    def read(self, file_path: Dict[str, str]):
        """
        read returns an iterable of instances that is directly
        iterated over when constructing vocab, and in the iterators.
        Since we will also pair this reader with a special iterator,
        we only have to worry about the case where the return value from
        this call is used to iterate for vocab creation.
        In addition, it is the return value from this that is passed
        into Trainer as the dataset (and then into the iterator)
        """
        datasets = {key: self.dataset_readers[key].read(fpath)
                    for key, fpath in file_path.items()}
        return MultitaskDataset(datasets, self.datasets_for_vocab_creation)


@DataIterator.register("multitask_iterator")
class MultiTaskDataIterator(DataIterator):
    def __init__(self,
                 iterators: Dict[str, DataIterator],
                 names_to_index: List[str],
                 iterate_forever: bool = False,
                 steps_per_epoch: int = None,
                 sampling_rates: List[float] = None) -> None:
        self.iterators = iterators
        self.names_to_index = names_to_index
        self.sampling_rates = sampling_rates
        self.iterate_forever = iterate_forever
        self.steps_per_epoch = steps_per_epoch

    def __call__(self,
                 multitask_dataset: MultitaskDataset,
                 num_epochs: int = None,
                 shuffle: bool = True):

        # get the number of batches in each of the sub-iterators for
        # the sampling rate
        num_batches_per_iterator = []
        for name in self.names_to_index:
            dataset = multitask_dataset.datasets[name]
            num_batches_per_iterator.append(
                self.iterators[name].get_num_batches(dataset)
            )

        total_batches_per_epoch = sum(num_batches_per_iterator)

        # make the sampling rates --
        p = np.array(num_batches_per_iterator, dtype=np.float) \
                                                / total_batches_per_epoch

        if self.iterate_forever:
            total_batches_per_epoch = 1000000000

        if self.steps_per_epoch:
            total_batches_per_epoch = self.steps_per_epoch

        if self.sampling_rates is not None:
            p = np.array(self.sampling_rates, dtype=np.float)

        for epoch in range(num_epochs):
            generators = []
            for name in self.names_to_index:
                dataset = multitask_dataset.datasets[name]
                generators.append(
                    self.iterators[name](
                        dataset,
                        num_epochs=1,
                        shuffle=shuffle,
                    )
                )

            n_batches_this_epoch = 0
            all_indices = np.arange(len(generators)).tolist()
            while n_batches_this_epoch < total_batches_per_epoch:
                index = np.random.choice(len(generators), p=p)
                try:
                    batch = next(generators[index])
                except StopIteration:
                    # remove this generator from the pile!
                    del generators[index]
                    if len(generators) == 0:
                        # something went wrong
                        raise ValueError
                    del all_indices[index]
                    newp = np.concatenate([p[:index], p[index+1:]])
                    newp /= newp.sum()
                    p = newp
                    continue

                # add the iterator id
                batch['dataset_index'] = torch.tensor(all_indices[index])
                yield batch

                n_batches_this_epoch += 1

    def _take_instances(self, *args, **kwargs):
        raise NotImplementedError

    def _memory_sized_lists(self, *args, **kwargs):
        raise NotImplementedError

    def _ensure_batch_is_sufficiently_small(self, *args, **kwargs):
        raise NotImplementedError

    def get_num_batches(self, multitask_dataset: MultitaskDataset) -> int:
        num_batches = 0
        for name, dataset in multitask_dataset.datasets.items():
            num_batches += self.iterators[name].get_num_batches(dataset)
        return num_batches

    def index_with(self, vocab: Vocabulary):
        for iterator in self.iterators.values():
            iterator.index_with(vocab)