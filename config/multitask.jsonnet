local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));

{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
            "roc_lm": {
                "type": "roc_lm",

            },
            "roc_hierarchy": {
                "type": "roc_hierarchy",

            },
            "atomic": {
                "type": "atomic",

            },
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["roc_lm", "roc_hierarchy", "atomic"],
   "iterate_forever": false,
   "iterators": {
       "roc_lm": {
            "type": "basic",
            "batch_size": 16
       },
       "roc_hierarchy": {
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
        "roc_hierarchy": dataset_root + "/ROCStories/ROCStories_winter2017 - ROCStories_winter2017.csv",
        "atomic": dataset_root + "/atomic/v4_atomic_trn.csv",
  },
  "validation_data_path": {
        "roc_lm": dataset_root + "/ROCStories/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv",
        "roc_hierarchy": dataset_root + "/ROCStories/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv",
        "atomic": dataset_root + "/atomic/v4_atomic_dev.csv",
  },
  "model": {
    "type": "knowledgeable_stories",
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
    "grad_norm": 2.0,
    "shuffle": true,
    "summary_interval": 500,
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
      "factor": 0.5,
      "patience": 0
    }
  }
}