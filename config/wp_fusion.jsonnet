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
local LR_RATE = std.parseJson(std.extVar("LR_RATE"));
local MOMENTUM = std.parseJson(std.extVar("MOMENTUM"));
local PATIENCE = std.parseInt(std.extVar("PATIENCE"));
local TRAINING_ITERATION_SIZE = std.parseInt(std.extVar("TRAINING_ITERATION_SIZE"));
local VALIDATION_ITERATION_SIZE = std.parseInt(std.extVar("VALIDATION_ITERATION_SIZE"));
local LR_PATIENCE = std.parseInt(std.extVar("LR_PATIENCE"));
local LR_REDUCE_RATE = std.parseJson(std.extVar("LR_REDUCE_RATE"));

{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
            "writing_prompts_hierarchy": {
                "type": "writing_prompts_hierarchy",
                "lazy": true,
                "batch_size" : 50,
            }
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_hierarchy"],
   "iterate_forever": false,
   "batches_per_epoch": TRAINING_ITERATION_SIZE,
   "sampling_rates": [1.0],
   "iterators": {
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
    },
  },
  "validation_iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_hierarchy"],
   "iterate_forever": false,
   "batches_per_epoch": VALIDATION_ITERATION_SIZE,
   "sampling_rates": [1.0],
   "iterators": {
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
    },
  },
  "train_data_path": {
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/train.wp_target",
  },
  "validation_data_path": {
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/valid.wp_target",
  },
  "model": {
    "type": "know_stories",
    "sentence_detach": false,
    "embedder_vocab_size": embedder_vocab_size,
    "lm_device": 1,
    "lm_finetune_final_layer_only": true,
    "dataset_config": {
        "writing_prompts_lm": {},
        "writing_prompts_hierarchy": {},
    },
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
    "sentence_autoencoder": {
        "input_dim": 768,
        "embedding_dim": 48,
        "hidden_dims": [384, 192, 96],
        "negative_slope": 0.1
    },
    "passage_autoencoder": {
        "input_dim": 768,
        "embedding_dim": 48,
        "hidden_dims": [384, 192, 96],
        "negative_slope": 0.1
    },
    "fusion_dense": {
        "input_dim": 1536,
        "num_layers": 2,
        "hidden_dims": 768,
        "activations": "elu",
        "dropout": 0.0
    }
  },
  "trainer": {
    "num_epochs": EPOCHS,
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