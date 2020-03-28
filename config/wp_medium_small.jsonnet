local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_GPUS = std.parseInt(std.extVar("NUM_GPUS"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));
local PASSAGE_BASE_BATCH_SIZE = 2;
local LM_BASE_BATCH_SIZE = 1;
local MAX_INSTANCES_IN_MEMORY = 64;

{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
             "writing_prompts_lm": {
                "type": "writing_prompts_lm",
                "lazy": true,
                "batch_size" : 10,
                "max_sentence_grouping": 10,
                "max_token_len": 256,

            },
            "writing_prompts_hierarchy": {
                "type": "writing_prompts_hierarchy",
                "lazy": true,
                "batch_size" : 100,
            }
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy"],
   "iterate_forever": false,
   "batches_per_epoch": 1000,
   "sampling_rates": [0.5, 0.5],
   "iterators": {
       "writing_prompts_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
    },
  },
  "validation_iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy"],
   "iterate_forever": false,
   "batches_per_epoch": 100,
   "sampling_rates": [0.5, 0.5],
   "iterators": {
       "writing_prompts_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
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
    "type": "know_stories",
    "lm_name": "gpt2-medium",
    "embedder_vocab_size": embedder_vocab_size,
    "dataset_config": {
        "writing_prompts_lm": {},
        "writing_prompts_hierarchy": {},
    },
    "sentence_seq2vec_encoder": {
      "type": "lstm",
      "input_size": 1024,
      "hidden_size": 1024,
      "num_layers": 3,
      "dropout": 0.0,
    },
    "passage_seq2seq_encoder": {
      "type": "lstm",
      "input_size": 1024,
      "hidden_size": 1024,
      "num_layers": 4,
      "dropout": 0.0,
    },
    "sentence_autoencoder": {
        "input_dim": 1024,
        "embedding_dim": 64,
        "hidden_dims":  [512, 256, 128],
        "negative_slope": 0.1
    },
    "passage_autoencoder": {
        "input_dim": 1024,
        "embedding_dim": 64,
        "hidden_dims": [512, 256, 128],
        "negative_slope": 0.1
    }
  },
  "trainer": {
    "num_epochs": 5,
    "validation_metric": "-loss",
    "patience": 2,
    "grad_norm": 5.0,
    "shuffle": false,
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
      "factor": 0.25,
 "patience": 1
    }
  }
}