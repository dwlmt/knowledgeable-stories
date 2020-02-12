local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.extVar("EMBEDDER_VOCAB_SIZE");

{
  "dataset_reader": {
    "type": "atomic"
  },
  "train_data_path": dataset_root + "/atomic/v4_atomic_trn.csv",
  "validation_data_path":  dataset_root + "/atomic/v4_atomic_dev.csv",
  "test_data_path":  dataset_root + "/atomic/v4_atomic_tst.csv",
  "model": {
    "type": "knowledgeable_stories",
    "embedder_vocab_size": 50268
  },
  "iterator": {
    "type": "basic",
    "batch_size": 16
  },
  "trainer": {
    "num_epochs": 50,
    "validation_metric": "-loss",
    "patience": 3,
    "grad_norm": 2.0,
    "shuffle": true,
    "cuda_device": [
      0
    ],
    "model_save_interval": 7200.0,
    "num_serialized_models_to_keep": 2,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01,
      "momentum": 0.9,
      "nesterov": true
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "patience": 0
    }
  }
}