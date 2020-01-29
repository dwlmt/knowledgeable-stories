local pwd = std.extVar("PWD");
local pwd2 = std.strReplace(pwd,'model','');
local train_path = pwd2 + "/fixtures/data/atomic_small.csv";

{
  "dataset_reader": {
    "type": "atomic"
  },
  "train_data_path": train_path ,
  "validation_data_path":  train_path,
  "model": {
    "type": "knowledgeable_stories",
    "embedder_vocab_size": 50267,
  },
  "iterator": {
    "type": "basic",
    "batch_size": 2
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device": -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    }
  }
}