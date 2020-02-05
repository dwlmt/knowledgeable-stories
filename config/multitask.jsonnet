local dataset_root = std.extVar("DATASET_ROOT");
{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
            "roc_lm": {
                "type": "roc_lm",

            },
            "atomic": {
                "type": "atomic",

            },
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["roc_lm","atomic"],
   "iterate_forever": false,
   "iterators": {
       "roc_lm": {
            "type": "basic",
            "batch_size": 16
       },
       "atomic": {
            "type": "basic",
            "batch_size": 16
       },
    },
  },
  "train_data_path": {
        "roc_lm": dataset_root + "/ROCStories/ROCStories_winter2017 - ROCStories_winter2017.csv",
        "atomic": dataset_root + "/atomic/v4_atomic_trn.csv",
  },
  "validation_data_path": {
        "roc_lm": dataset_root + "/ROCStories/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv",
        "atomic": dataset_root + "/atomic/v4_atomic_dev.csv",
  },
  "model": {
    "type": "knowledgeable_stories",
    "embedder_vocab_size": 50268
  },
  "trainer": {
    "num_epochs": 50,
    "validation_metric": "-loss",
    "patience": 3,
    "grad_norm": 2.0,
    "shuffle": true,
    "summary_interval": 500,
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