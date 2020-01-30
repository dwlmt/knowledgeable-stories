local dataset_root = std.extVar("DATASET_ROOT");

{
  "dataset_reader": {
    "type": "atomic"
  },
  "train_data_path": dataset_root + "/atomic/v4_atomic_small.csv",
  "validation_data_path":  dataset_root + "/atomic/v4_atomic_small.csv",
  "test_data_path":  dataset_root + "/atomic/v4_atomic_small.csv",
  "evaluate_on_test": true,
  "model": {
    "type": "knowledgeable_stories",
    "embedder_vocab_size": 50268
  },
  "iterator": {
    "type": "basic",
    "batch_size": 24
  },
  "trainer": {
    "num_epochs": 20,
    "validation_metric": "-loss",
    "patience": 1,
    "shuffle": true,
    "cuda_device": [
      0, 1
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
      "mode": "max",
      "patience": 2
    }
  }
}