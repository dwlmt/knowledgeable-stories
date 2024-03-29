local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_GPUS = std.parseInt(std.extVar("NUM_GPUS"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));
local PASSAGE_BASE_BATCH_SIZE = 1;
local LM_BASE_BATCH_SIZE = 1;
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
                "slide" : 1.0,
            }
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy"],
   "iterate_forever": false,
   "batches_per_epoch": TRAINING_ITERATION_SIZE,
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
   "batches_per_epoch": VALIDATION_ITERATION_SIZE,
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
    "lm_device": 1,
    "tdvae_device": 2,
    "lm_finetune_final_layer_only": false,
    "sent_offsets": [-3, -2, -1, 1, 2, 3],
    "sent_scales": [2.5, 5.0, 10.0, 10.0, 5.0, 2.5],
    "label_smoothing": 0.0,
    "embedder_vocab_size": embedder_vocab_size,
    "dataset_config": {
        "writing_prompts_lm": {},
        "writing_prompts_hierarchy": {},
    },
    "loss_weights" : {
        "lm_loss": 1.0,
        "tdvae_loss": 1.0,
        "sentence_disc_loss": 1.0,
        "sentence_autoencoder": 1.0,
        "position_loss": 1.0,
        "sentiment_loss": 1.0,
    },
    "sentence_seq2vec_encoder": {
      "type": "seq2seq_pooler",
      "pooler": {
        "type": "final_pooler",
        "embedding_dim": 1024
      },
      "seq2seq_encoder": {
        "type": "pytorch_transformer",
        "input_dim": 1024,
        "num_layers": 5,
        "num_attention_heads": 16,
        "positional_encoding": "embedding",
        "dropout_prob": 0.0,
      }
    },
    "sentence_2_seq2vec_encoder": {
      "type": "seq2seq_pooler",
      "pooler": {
       "type": "final_pooler",
       "embedding_dim": 1024,
      },
      "seq2seq_encoder": {
        "type": "pytorch_transformer",
        "input_dim": 1024,
        "num_layers": 5,
        "positional_encoding": "embedding",
        "num_attention_heads": 16,
        "dropout_prob": 0.0,
      }
    },
    "passage_tdvae": {
         "x_size": 2048,
         "input_size": 2048,
         "belief_size": 2048,
         "z_posterior_size": 2048,
         "num_layers": 6,
         "samples_per_seq": 100,
         "t_diff_min": 1,
         "t_diff_max": 5,
         "d_block_hidden_size": 512,
         "decoder_hidden_sizes": [6144, 4096, 2048],
    },
    "sentence_autoencoder": {
        "input_dim": 2048,
        "embedding_dim": 64,
        "hidden_dims":  [1024, 512, 256, 128],
        "negative_slope": 0.1
    },
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