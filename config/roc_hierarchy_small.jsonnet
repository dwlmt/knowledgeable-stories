local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.extVar("EMBEDDER_VOCAB_SIZE");

{
  "dataset_reader": {
    "type": "roc_hierarchy"
  },
  "train_data_path": dataset_root + "/ROCStories/roc_train_50.csv",
  "validation_data_path":  dataset_root + "/ROCStories/roc_val_50.csv",
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
  "iterator": {
    "type": "basic",
    "batch_size": 16
  },
  "trainer": {
    "num_epochs": 500,
    "validation_metric": "-loss",
    "patience": 1,
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
      "patience": 3
    }
  }
}