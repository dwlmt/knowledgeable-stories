local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_GPUS = std.parseInt(std.extVar("NUM_GPUS"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));
local PASSAGE_BASE_BATCH_SIZE = 2;
local LM_BASE_BATCH_SIZE = 1;
local KB_BASE_BATCH_SIZE = 4;
local MAX_INSTANCES_IN_MEMORY = std.parseInt(std.extVar("MAX_INSTANCES_IN_MEMORY"));
local EPOCHS = std.parseInt(std.extVar("EPOCHS"));
local PATIENCE = std.parseInt(std.extVar("PATIENCE"));
local LR_RATE = std.parseJson(std.extVar("LR_RATE"));
local MOMENTUM = std.parseJson(std.extVar("MOMENTUM"));
local LR_PATIENCE = std.parseInt(std.extVar("LR_PATIENCE"));
local LR_REDUCE_RATE = std.parseJson(std.extVar("LR_REDUCE_RATE"));
local TRAINING_ITERATION_SIZE = std.parseInt(std.extVar("TRAINING_ITERATION_SIZE"));
local VALIDATION_ITERATION_SIZE = std.parseInt(std.extVar("VALIDATION_ITERATION_SIZE"));

{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
              "writing_prompts_lm": {
                "type": "writing_prompts_lm",
                "lazy": true,
                "batch_size" : 36,
            "max_sentence_grouping": 36,
            "max_token_len": 768,
            },
            "writing_prompts_hierarchy": {
                "type": "writing_prompts_hierarchy",
                "lazy": true,
            "batch_size" : 100,
            },
            "atomic_lm" : {
                "type": "atomic"
            },
            "swag_know_lm" : {
                "type": "swag_know_lm"
            }
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy", "atomic_lm", "swag_know_lm"],
   "sampling_rates": [1.0 / 4.0 , 1.0 / 4.0 , 1.0 / 4.0 , 1.0 / 4.0 ],
   "iterate_forever": false,
   "batches_per_epoch": TRAINING_ITERATION_SIZE,
   "iterators": {
       "writing_prompts_lm": {
            "type": "basic",
            "batch_size":  LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "atomic_lm": {
            "type": "basic",
            "batch_size":  KB_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "swag_know_lm": {
            "type": "basic",
            "batch_size":  KB_BASE_BATCH_SIZE * NUM_GPUS,
             "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
    },
  },
  "validation_iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy", "atomic_lm", "swag_know_lm"],
   "sampling_rates": [1.0 / 4.0 , 1.0 / 4.0 , 1.0 / 4.0 , 1.0 / 4.0 ],
   "iterate_forever": false,
   "batches_per_epoch": VALIDATION_ITERATION_SIZE,
   "iterators": {
       "writing_prompts_lm": {
            "type": "basic",
            "batch_size":  LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "atomic_lm": {
            "type": "basic",
            "batch_size":  KB_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "swag_know_lm": {
            "type": "basic",
            "batch_size":  KB_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
    },
  },
  "train_data_path": {
        "writing_prompts_lm": dataset_root + "/WritingPrompts/train.wp_target",
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/train.wp_target",
        "atomic_lm": dataset_root + "/atomic/v4_atomic_trn.csv",
        "swag_know_lm": dataset_root + "/swagaf/data/train_full.csv",
  },
  "validation_data_path": {
        "writing_prompts_lm": dataset_root + "/WritingPrompts/valid.wp_target",
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/valid.wp_target",
        "atomic_lm": dataset_root + "/atomic/v4_atomic_dev.csv",
        "swag_know_lm": dataset_root + "/swagaf/data/val_full.csv",
  },
  "model": {
    "type": "know_stories",
    "lm_name": "gpt2-medium",
"lm_device": 1,
    "dataset_config": {
        "writing_prompts_lm": {},
        "writing_prompts_hierarchy": {},
        "roc_lm": {},
        "roc_hierarchy": {},
        "cmu_book_lm": {},
        "cmu_book_hierarchy": {},
        "cmu_movie_lm": {},
        "cmu_movie_hierarchy": {},
        "atomic_lm": {},
        "swag_know_lm": {},
    },
    "embedder_vocab_size": embedder_vocab_size,
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
      "num_layers": 5,
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
    "num_epochs": 10000,
    "validation_metric": "-loss",
 "patience": PATIENCE,
    "grad_norm": 5.0,
    "shuffle": false,
    "summary_interval": 500,
    "model_save_interval": 7200.0,
    "num_serialized_models_to_keep": 2,
    "cuda_device": if NUM_GPUS > 1 then std.range(0, NUM_GPUS - 1) else 0,
    "optimizer": {
      "type": "sgd",
      "lr": LR_RATE,
      "momentum": MOMENTUM,
      "nesterov": true
    },
    "learning_rate_scheduler": {
      "type": "reduce_on_plateau",
      "factor": LR_REDUCE_RATE,
"patience": LR_PATIENCE,
    }
  }
}