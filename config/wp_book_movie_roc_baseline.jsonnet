local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_GPUS = std.parseInt(std.extVar("NUM_GPUS"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));
local PASSAGE_BASE_BATCH_SIZE = 2;
local LM_BASE_BATCH_SIZE = 1;
local MAX_INSTANCES_IN_MEMORY = std.parseInt(std.extVar("MAX_INSTANCES_IN_MEMORY"));
local EPOCHS = std.parseInt(std.extVar("EPOCHS"));
local PATIENCE = std.parseInt(std.extVar("PATIENCE"));
local LR_RATE = std.parseJson(std.extVar("LR_RATE"));
local MOMENTUM = std.parseJson(std.extVar("MOMENTUM"));
local LR_PATIENCE = std.parseInt(std.extVar("LR_PATIENCE"));
local LR_REDUCE_RATE = std.parseInt(std.extVar("LR_REDUCE_RATE"));
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
                "batch_size" : 10,
                "max_sentence_grouping": 10,
                "max_token_len": 256,
            },
            "writing_prompts_hierarchy": {
                "type": "writing_prompts_hierarchy",
                "lazy": true,
                "batch_size" : 100,
            },
            "roc_lm": {
                "type": "roc_lm",
                "lazy": true,
            },
            "roc_hierarchy": {
                "type": "roc_hierarchy",
                "lazy": true,
            },
            "cmu_movie_lm": {
                "type": "cmu_movie_lm",
                "lazy": true,
                "batch_size" : 10,
                "max_sentence_grouping": 10,
                "max_token_len": 256,
            },
            "cmu_movie_hierarchy": {
                "type": "cmu_movie_hierarchy",
                "lazy": true,
                "batch_size" : 100,
            },
             "cmu_book_lm": {
                "type": "cmu_book_lm",
                "lazy": true,
                "batch_size" : 10,
                "max_sentence_grouping": 10,
                "max_token_len": 256,
            },
            "cmu_book_hierarchy": {
                "type": "cmu_book_hierarchy",
                "lazy": true,
                "batch_size" : 100,
            },
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy", "roc_lm", "roc_hierarchy",
   "cmu_book_lm", "cmu_book_hierarchy", "cmu_movie_lm", "cmu_movie_hierarchy"],
   "sampling_rates": [1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0],
   "iterate_forever": false,
   "batches_per_epoch": ,TRAINING_ITERATION_SIZE
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
       "cmu_movie_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "cmu_movie_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "cmu_book_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,

       },
       "cmu_book_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "roc_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "roc_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
    },
  },
  "validation_iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy", "roc_lm", "roc_hierarchy",
   "cmu_book_lm", "cmu_book_hierarchy", "cmu_movie_lm", "cmu_movie_hierarchy"],
   "sampling_rates": [1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0],
   "iterate_forever": false,
   "batches_per_epoch": , VALIDATION_ITERATION_SIZE
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
       "cmu_movie_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "cmu_movie_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "cmu_book_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,

       },
       "cmu_book_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "roc_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "roc_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
    },
  },
  "train_data_path": {
        "writing_prompts_lm": dataset_root + "/WritingPrompts/train.wp_target",
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/train.wp_target",
        "roc_lm": dataset_root + "/ROCStories/ROCStories_train_all.csv",
        "roc_hierarchy": dataset_root + "/ROCStories/ROCStories_train_all.csv",
        "cmu_movie_lm": dataset_root + "/MovieSummaries/own_processed/plot_summaries_train",
        "cmu_movie_hierarchy": dataset_root + "/MovieSummaries/own_processed/plot_summaries_train",
        "cmu_book_lm": dataset_root + "/booksummaries/booksummaries.txt",
        "cmu_book_hierarchy": dataset_root + "/booksummaries/booksummaries.txt",
  },
  "validation_data_path": {
        "writing_prompts_lm": dataset_root + "/WritingPrompts/valid.wp_target",
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/valid.wp_target",
        "roc_lm": dataset_root + "/ROCStories/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv",
        "roc_hierarchy":  dataset_root + "/ROCStories/cloze_test_val__winter2018-cloze_test_ALL_val - 1 - 1.csv",
        "cmu_movie_lm": dataset_root + "/MovieSummaries/own_processed/plot_summaries_valid",
        "cmu_movie_hierarchy": dataset_root + "/MovieSummaries/own_processed/plot_summaries_valid",
        "cmu_book_lm": dataset_root + "/booksummaries/booksummaries.txt",
        "cmu_book_hierarchy": dataset_root + "/booksummaries/booksummaries.txt",
  },
  "model": {
    "type": "know_stories",
    "embedder_vocab_size": embedder_vocab_size,
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