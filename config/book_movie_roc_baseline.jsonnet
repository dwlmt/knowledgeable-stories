local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_GPUS = std.parseInt(std.extVar("NUM_GPUS"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));
local PASSAGE_BASE_BATCH_SIZE = 2;
local LM_BASE_BATCH_SIZE = 1;

{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
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
            },
            "cmu_movie_hierarchy": {
                "type": "cmu_movie_hierarchy",
                "lazy": true,
            },
             "cmu_book_lm": {
                "type": "cmu_book_lm",
                "lazy": true,
            },
            "cmu_book_hierarchy": {
                "type": "cmu_book_hierarchy",
                "lazy": true,
            },

        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["roc_lm", "roc_hierarchy",
   "cmu_book_lm", "cmu_book_hierarchy", "cmu_movie_lm", "cmu_movie_hierarchy"],
   "iterate_forever": false,
   "batches_per_epoch": 10000,
   "iterators": {
       "cmu_movie_lm": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
       },
       "cmu_movie_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
       },
       "cmu_book_lm": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
       },
       "cmu_book_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
       },
       "roc_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
       },
       "roc_hierarchy": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
       },
    },
  },
  "validation_iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["roc_lm", "roc_hierarchy",
   "cmu_book_lm", "cmu_book_hierarchy", "cmu_movie_lm", "cmu_movie_hierarchy"],
   "iterate_forever": false,
   "batches_per_epoch": 1000,
   "iterators": {
       "cmu_movie_lm": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
       },
       "cmu_movie_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
       },
       "cmu_book_lm": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
       },
       "cmu_book_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
       },
        "roc_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
       },
       "roc_hierarchy": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
       },
    },
  },
  "train_data_path": {
        "roc_lm": dataset_root + "/ROCStories/ROCStories_train_all.csv",
        "roc_hierarchy": dataset_root + "/ROCStories/ROCStories_train_all.csv",
        "cmu_movie_lm": dataset_root + "/MovieSummaries/own_processed/plot_summaries_train",
        "cmu_movie_hierarchy": dataset_root + "/MovieSummaries/own_processed/plot_summaries_train",
        "cmu_book_lm": dataset_root + "/booksummaries/booksummaries.txt",
        "cmu_book_hierarchy": dataset_root + "/booksummaries/booksummaries.txt",
  },
  "validation_data_path": {
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