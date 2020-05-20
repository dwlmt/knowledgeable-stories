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

local WP_TRAIN_PATH = dataset_root + "/WritingPrompts/train.wp_target";
local WP_VALID_PATH = dataset_root + "/WritingPrompts/valid.wp_target";

{
  "dataset_reader": {
    "type": "interleaving",
    "readers": {
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
        "batch_size" : 200,
        "slide" : 1.0,
    }
    }
  },
  "train_data_path":  ||| {{"writing_prompts_lm": "%(WP_TRAIN_PATH)", "writing_prompts_hierarchy": "%(WP_TRAIN_PATH)"}}||| % self,
  "validation_data_path": ||| {{"writing_prompts_lm": "%(WP_VALID_PATH)","writing_prompts_hierarchy": "%(WP_VALID_PATH)"}}||| % self,
  "data_loader": {
    "batch_size": 1,
    "batches_per_epoch": TRAINING_ITERATION_SIZE,
    "num_workers": 6,
  },
  "validation_data_loader": {
    "batch_size": 1,
    "batches_per_epoch": VALIDATION_ITERATION_SIZE,
    "num_workers": 6,
  },
  "model": {
    "type": "know_stories",
    "lm_name": "gpt2-medium",
    "lm_device": 1,
    "lm_finetune_final_layer_only": false,
    "sent_offsets": [-1, 1],
    "sent_scales": [10.0, 10.0],
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
        "num_layers": 3,
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
        "num_layers": 3,
        "positional_encoding": "embedding",
        "num_attention_heads": 16,
        "dropout_prob": 0.0,
      }
    },
    "sentiment_dense": {
        "input_dim": 2048,
        "num_layers": 1,
        "hidden_dims": 1,
        "activations": "linear",
        "dropout": 0.0
    },
    "position_dense": {
        "input_dim": 2048,
        "num_layers": 1,
        "hidden_dims": 1,
        "activations": "linear",
        "dropout": 0.0
    },
    "passage_tdvae": {
         "x_size": 2048,
         "input_size": 2048,
         "belief_size": 1024,
         "z_posterior_size": 1024,
         "num_layers": 5,
         "samples_per_seq": 200,
         "t_diff_min": 1,
         "t_diff_max": 8,
         "d_block_hidden_size": 320,
         "decoder_hidden_size": 1024,
    },
    "sentence_autoencoder": {
        "input_dim": 2048,
        "embedding_dim": 64,
        "hidden_dims":  [1024, 512, 256, 128],
        "negative_slope": 0.1
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