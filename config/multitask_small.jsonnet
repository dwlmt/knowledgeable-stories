local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_GPUS = std.parseInt(std.extVar("NUM_GPUS"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));

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
        "roc_lm": dataset_root + "/ROCStories/roc_train_50.csv",
        "atomic": dataset_root + "/atomic/v4_atomic_small.csv",
  },
  "validation_data_path": {
        "roc_lm": dataset_root + "/ROCStories/roc_val_50.csv",
        "atomic": dataset_root + "/atomic/v4_atomic_small.csv",
  },
  "model": {
    "type": "know_stories",
    "embedder_vocab_size": embedder_vocab_size,
    "sentence_seq2vec_encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": 768,
      "num_layers": 2,
      "dropout": 0.0,
    },
    "passage_seq2seq_encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": 768,
      "num_layers": 2,
      "dropout": 0.0,
    },
  },
  "trainer": {
    "num_epochs": 50,
    "validation_metric": "-loss",
    "patience": 3,
    "grad_norm": 5.0,
    "shuffle": true,
    "model_save_interval": 7200.0,
    "num_serialized_models_to_keep": 2,
    "cuda_device": if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01,
      "momentum": 0.9,
      "nesterov": true
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.1,
      "patience": 0
    }
  }
}