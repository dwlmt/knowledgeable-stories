local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");

{
  "dataset_reader": {
    "type": "writing_prompts_hierarchy"
  },
  "train_data_path": dataset_root + "/WritingPrompts/writing_prompts_25",
  "validation_data_path":  dataset_root + "/WritingPrompts/writing_prompts_25",
  "model": {
    "type": "knowledgeable_stories",
    "embedder_vocab_size": 50268,
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
      "num_layers": 4,
      "dropout": 0.0,
    },
  },
  "iterator": {
    "type": "basic",
    "batch_size": 2
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