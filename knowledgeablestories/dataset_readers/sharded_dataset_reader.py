import glob
import logging
from typing import Iterable

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance

logger = logging.getLogger(__name__)

@DatasetReader.register("sharded_simple")
class ShardedDatasetReader(DatasetReader):
    """
    Wraps another dataset reader and uses it to read from multiple input files.

    This is a simplified version of the new head reader to work around the multi-processing problems.
    """
    def __init__(self, base_reader: DatasetReader, lazy= False) -> None:
        super().__init__(lazy=lazy)

        self.reader = base_reader

    def text_to_instance(self, *args) -> Instance:
        """
        Just delegate to the base reader text_to_instance.
        """
        return self.reader.text_to_instance(*args)  # type: ignore

    def _read(self, file_path: str) -> Iterable[Instance]:
        shards = glob.glob(file_path)
        # Ensure a consistent order.
        shards.sort()

        for i, shard in enumerate(shards):
            logger.info(f"reading instances from {shard}")
            for instance in self.reader.read(shard):
                yield instance