local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_GPUS = std.parseInt(std.extVar("NUM_GPUS"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));
local PASSAGE_BASE_BATCH_SIZE = 2;
local LM_BASE_BATCH_SIZE = 1;
local KB_BASE_BATCH_SIZE = 4;
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
            },
            "roc_lm": {
                "type": "roc_lm",
                "lazy": true,
                "batch_size" : 10,
                "max_sentence_grouping": 10,
                "max_token_len": 256,
            },
            "roc_hierarchy": {
                "type": "roc_hierarchy",
                "lazy": true,
                "batch_size" : 100,
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
            "atomic" : {
                "type": "atomic"
            },
            "swag_know_lm" : {
                "type": "swag_know_lm"
            },
            "schmoop_lm": {
                "type": "multiprocess_unreleased",
                "base_reader": {
                    "type": "multifile_lm",
                },
                "num_workers": 1,
            },
            "schmoop_hierarchy": {
                "type": "multiprocess_unreleased",
                "base_reader": {
                    "type": "multifile_lm",
                },
                "num_workers": 1,
            },
            "bookscorpus_lm": {
                "type": "multiprocess_unreleased",
                "base_reader": {
                    "type": "multifile_lm",
                },
                "num_workers": 1,
            },
            "bookscorpus_hierarchy": {
                "type": "multiprocess_unreleased",
                "base_reader": {
                    "type": "multifile_hierarchy",
                },
                "num_workers": 1,
            },
            "filmcorpus_lm": {
                "type": "multiprocess_unreleased",
                "base_reader": {
                    "type": "multifile_lm",
                },
                "num_workers": 1,
            },
            "filmcorpus_hierarchy": {
                "type": "multiprocess_unreleased",
                "base_reader": {
                    "type": "multifile_hierarchy",
                },
                "num_workers": 1,
            }
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy", "roc_lm", "roc_hierarchy",
   "cmu_book_lm", "cmu_book_hierarchy", "cmu_movie_lm", "cmu_movie_hierarchy", "atomic", "swag_know_lm",
   "schmoop_lm", "schmoop_hierarchy", "bookscorpus_lm", "bookscorpus_hierarchy",  "filmcorpus_lm", "filmcorpus_hierarchy"],
   "iterate_forever": false,
   "batches_per_epoch": 100000,
   "sampling_rates":  [1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0],
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
       "atomic": {
            "type": "basic",
            "batch_size":  KB_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "swag_know_lm": {
            "type": "basic",
            "batch_size":  KB_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "schmoop_lm": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "schmoop_hierarchy": {
            "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "bookscorpus_lm": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "bookscorpus_hierarchy": {
            "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "filmcorpus_lm": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "filmcorpus_hierarchy": {
            "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
    },
  },
  "validation_iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy", "roc_lm", "roc_hierarchy",
   "cmu_book_lm", "cmu_book_hierarchy", "cmu_movie_lm", "cmu_movie_hierarchy", "atomic", "swag_know_lm",
   "schmoop_lm", "schmoop_hierarchy", "bookscorpus_lm", "bookscorpus_hierarchy", "filmcorpus_lm", "filmcorpus_hierarchy"],
   "iterate_forever": false,
   "batches_per_epoch": 10000,
   "sampling_rates":  [1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0, 1.0 / 16.0],
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
       "atomic": {
            "type": "basic",
            "batch_size":  KB_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "swag_know_lm": {
            "type": "basic",
            "batch_size":  KB_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "schmoop_lm": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "schmoop_hierarchy": {
            "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "bookscorpus_lm": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "bookscorpus_hierarchy": {
            "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "filmcorpus_lm": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
       },
       "filmcorpus_hierarchy": {
            "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": 1,
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
        "atomic": dataset_root + "/atomic/v4_atomic_trn.csv",
        "swag_know_lm": dataset_root + "/swagaf/data/train_full.csv",
        "schmoop_lm": dataset_root + "/schmoop/stories//*//*",
        "schmoop_hierarchy": dataset_root + "/schmoop/stories//*//*",
        "bookscorpus_lm": dataset_root + "/WritingPrompts/BooksCorpus/*",
        "bookscorpus_hierarchy": dataset_root + "/WritingPrompts/BooksCorpus/*",
        "filmscorpus_lm": dataset_root + "/WritingPrompts/FilmsCorpus/imsdb_raw_nov_2015/imsdb_raw_nov_2015/*",
        "filmscorpus_hierarchy": dataset_root + "/WritingPrompts/FilmsCorpus/imsdb_raw_nov_2015/imsdb_raw_nov_2015/*",
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
        "swag_know_lm": dataset_root + "/swagaf/data/val_full.csv",
        "schmoop_lm": dataset_root + "/schmoop/stories//*//*",
        "schmoop_hierarchy": dataset_root + "/schmoop/stories//*//*",
        "bookscorpus_lm": dataset_root + "/WritingPrompts/BooksCorpus/*",
        "bookscorpus_hierarchy": dataset_root + "/WritingPrompts/BooksCorpus/*",
        "filmscorpus_lm": dataset_root + "/WritingPrompts/FilmsCorpus/imsdb_raw_nov_2015/imsdb_raw_nov_2015/*",
        "filmscorpus_hierarchy": dataset_root + "/WritingPrompts/FilmsCorpus/imsdb_raw_nov_2015/imsdb_raw_nov_2015/*",
  },
  "model": {
    "type": "know_stories",
     "dataset_config": {
        "writing_prompts_lm": {},
        "writing_prompts_hierarchy": {},
        "roc_lm": {},
        "roc_hierarchy": {},
        "cmu_book_lm": {},
        "cmu_book_hierarchy": {},
        "cmu_movie_lm": {},
        "cmu_movie_hierarchy": {},
        "atomic": {},
        "swag_know_lm": {},
        "schmoop_lm": {},
        "schmoop_hierarchy": {},
        "bookscorpus_lm": {},
        "bookscorpus_hierarchy": {},
        "filmcorpus_lm" : {},
        "filmcorpus_hierarchy": {},
    },
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
    "num_epochs": 100,
    "validation_metric": "-loss",
    "patience": 2,
    "grad_norm": 5.0,
    "shuffle": true,
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
      "factor": 0.1,
      "patience": 1
    }
  }
}