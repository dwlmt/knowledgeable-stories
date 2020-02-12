local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
             "writing_prompts_lm": {
                "type": "writing_prompts_lm",
                "cache_directory": dataset_cache_root + "/baseline/writing_prompts_lm/"

            },
            "writing_prompts_hierarchy": {
                "type": "writing_prompts_hierarchy",
                  "cache_directory": dataset_cache_root + "/baseline/writing_prompts_hierarchy/"

            }
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy"],
   "iterate_forever": false,
   "iterators": {
       "writing_prompts_lm": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 10000,
       },
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 10000,
       },
    },
  },
  "validation_iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy"],
   "iterate_forever": false,
   "iterators": {
       "writing_prompts_lm": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 2000,
       },
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 2000,
       },
    },
  },
  "train_data_path": {
        "writing_prompts_lm": dataset_root + "/WritingPrompts/train.wp_target",
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/train.wp_target",
  },
  "validation_data_path": {
        "writing_prompts_lm": dataset_root + "/WritingPrompts/valid.wp_target",
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/valid.wp_target",
  },
  "model": {
    "type": "knowledgeable_stories",
    "embedder_vocab_size": 50268,
    "sentence_seq2vec_encoder": {
      "type": "lstm",
      "input_size": 768,
      "hidden_size": 768,
      "num_layers": 3,
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
  "trainer": {
    "num_epochs": 1000,
    "validation_metric": "-loss",
    "patience": 2,
    "grad_norm": 2.0,
    "shuffle": true,
    "summary_interval": 500,
    "cuda_device": [
      0, 1, 2, 3
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
      "patience": 1
    }
  }
}