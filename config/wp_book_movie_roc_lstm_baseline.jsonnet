local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.extVar("EMBEDDER_VOCAB_SIZE");

{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
            "writing_prompts_lm": {
                "type": "writing_prompts_lm",
                #"cache_directory": dataset_cache_root + "/baseline/writing_prompts_lm/"

            },
            "writing_prompts_hierarchy": {
                "type": "writing_prompts_hierarchy",
                #"cache_directory": dataset_cache_root + "/baseline/writing_prompts_hierarchy/"

            },
            "roc_lm": {
                "type": "roc_lm",
                #"cache_directory": dataset_cache_root + "/baseline/roc_lm/"
            },
            "roc_hierarchy": {
                "type": "roc_hierarchy",
                #"cache_directory": dataset_cache_root + "/baseline/roc_hierarchy/"
            },
            "cmu_movie_lm": {
                "type": "cmu_movie_lm"
            },
            "cmu_movie_hierarchy": {
                "type": "cmu_movie_hierarchy"
            },
             "cmu_book_lm": {
                "type": "cmu_book_lm"
            },
            "cmu_book_hierarchy": {
                "type": "cmu_book_hierarchy"
            },

        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy", "roc_lm", "roc_hierarchy"],
   "iterate_forever": false,
   "iterators": {
       "writing_prompts_lm": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 5000,
       },
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 5000,
       },
       "cmu_movie_lm": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 2500,
       },
       "cmu_movie_hierarchy": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 2500,
       },
       "cmu_book_lm": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 2500,
       },
       "cmu_book_hierarchy": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 2500,
       },
       "roc_lm": {
            "type": "basic",
            "batch_size": 32,
            "instances_per_epoch": 2500,
       },
       "roc_hierarchy": {
            "type": "basic",
            "batch_size": 32,
            "instances_per_epoch": 2500,
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
            "instances_per_epoch": 500,
       },
       "writing_prompts_hierarchy": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 500,
       },
       "roc_lm": {
            "type": "basic",
            "batch_size": 32,
            "instances_per_epoch": 250,
       },
       "roc_hierarchy": {
            "type": "basic",
            "batch_size": 32,
            "instances_per_epoch": 250,
       },
        "cmu_movie_lm": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 250,
       },
       "cmu_movie_hierarchy": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 250,
       },
       "cmu_book_lm": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 250,
       },
       "cmu_book_hierarchy": {
            "type": "basic",
            "batch_size": 8,
            "instances_per_epoch": 250,
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
        "roc_lm": dataset_root + "/ROCStories/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv",
        "roc_hierarchy":  dataset_root + "/ROCStories/cloze_test_test__winter2018-cloze_test_ALL_test - 1.csv",
        "cmu_movie_lm": dataset_root + "/MovieSummaries/own_processed/plot_summaries_valid",
        "cmu_movie_hierarchy": dataset_root + "/MovieSummaries/own_processed/plot_summaries_valid",
        "cmu_book_lm": dataset_root + "/booksummaries/booksummaries.txt",
        "cmu_book_hierarchy": dataset_root + "/booksummaries/booksummaries.txt",
  },
  "model": {
    "type": "knowledgeable_stories",
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