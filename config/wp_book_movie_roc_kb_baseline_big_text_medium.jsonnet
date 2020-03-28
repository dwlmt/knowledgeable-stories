local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_GPUS = std.parseInt(std.extVar("NUM_GPUS"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));
local PASSAGE_BASE_BATCH_SIZE = 2;
local LM_BASE_BATCH_SIZE = 1;
local KB_BASE_BATCH_SIZE = 4;
local MAX_INSTANCES_IN_MEMORY = std.parseInt(std.extVar("MAX_INSTANCES_IN_MEMORY"));
local EPOCHS = std.parseInt(std.extVar("EPOCHS");
local PATIENCE = std.parseInt(std.extVar("PATIENCE");
local LR_RATE = std.parseJson(std.extVar("LR_RATE");
local MOMENTUM = std.parseJson(std.extVar("MOMENTUM");
local LR_PATIENCE = std.parseInt(std.extVar("LR_PATIENCE");
local LR_REDUCE_RATE = std.parseInt(std.extVar("LR_REDUCE_RATE");
local TRAINING_ITERATION_SIZE = std.parseInt(std.extVar("TRAINING_ITERATION_SIZE");
local VALIDATION_ITERATION_SIZE = std.parseInt(std.extVar("VALIDATION_ITERATION_SIZE");

{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
        "writing_prompts_lm": {
          "type": "writing_prompts_lm",
          "lazy": true,
          "batch_size" : 10,
          "max_sentence_grouping": 14,
          "max_token_len": 384,
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
            "max_sentence_grouping": 14,
            "max_token_len": 384,
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
            "max_sentence_grouping": 14,
            "max_token_len": 384,
        },
        "cmu_book_hierarchy": {
            "type": "cmu_book_hierarchy",
            "lazy": true,
            "batch_size" : 100,
        },
        "atomic_lm" : {
            "type": "atomic",
            "lazy": true,
        },
        "swag_know_lm" : {
            "type": "swag_know_lm",
             "lazy": true,
        },
         "cbt_lm": {
           "type": "cbt_lm",
           "lazy": true,
           "batch_size" : 10,
           "max_sentence_grouping": 10,
           "max_token_len": 256,
        },
        "cbt_hierarchy": {
            "type": "cbt_hierarchy",
            "lazy": true,
            "batch_size" : 100,
        },
        "schmoop_lm": {
            "type": "sharded_simple",
            "lazy": true,
            "base_reader": {
                "type": "multifile_lm",
                "lazy": true,
            },
        },
        "schmoop_hierarchy": {
            "type": "sharded_simple",
            "lazy": true,
            "base_reader": {
                "type": "multifile_lm",
                "lazy": true,
            },
        },
        "bookscorpus_lm": {
            "type": "sharded_simple",
            "lazy": true,
            "base_reader": {
                "type": "multifile_lm",
                "lazy": true,
            },
        },
        "bookscorpus_hierarchy": {
            "type": "sharded_simple",
             "lazy": true,
            "base_reader": {
                "type": "multifile_hierarchy",
                "lazy": true,
            },
        },
        "filmcorpus_lm": {
            "type": "sharded_simple",
            "lazy": true,
            "base_reader": {
                "type": "multifile_lm",
                "lazy": true,
            },
        },
        "filmcorpus_hierarchy": {
            "type": "sharded_simple",
            "lazy": true,
            "base_reader": {
                "type": "multifile_hierarchy",
                "lazy": true,
            },
        },
    },
  },
  "iterator": {
   "type": "multitask_iterator",
    "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy", "roc_lm", "roc_hierarchy",
      "cmu_book_lm", "cmu_book_hierarchy", "cbt_lm", "cbt_hierarchy","cmu_movie_lm", "cmu_movie_hierarchy", "atomic_lm", "swag_know_lm",
       "schmoop_lm", "schmoop_hierarchy", "bookscorpus_lm", "bookscorpus_hierarchy", "filmcorpus_lm", "filmcorpus_hierarchy"],
       "iterate_forever": false,
       "batches_per_epoch": ,TRAINING_ITERATION_SIZE
       "sampling_rates":  [1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0],
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
     "cbt_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "cbt_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "schmoop_lm": {
           "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "schmoop_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY
       },
       "bookscorpus_lm": {
           "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "bookscorpus_hierarchy": {
           "type": "basic",
           "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
           "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "filmcorpus_lm": {
           "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "filmcorpus_hierarchy": {
           "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
    },
  },
  "validation_iterator": {
   "type": "multitask_iterator",
    "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy", "roc_lm", "roc_hierarchy",
      "cmu_book_lm", "cmu_book_hierarchy", "cbt_lm", "cbt_hierarchy","cmu_movie_lm", "cmu_movie_hierarchy", "atomic_lm", "swag_know_lm",
      "schmoop_lm", "schmoop_hierarchy", "bookscorpus_lm", "bookscorpus_hierarchy", "filmcorpus_lm", "filmcorpus_hierarchy"],
      "iterate_forever": false,
      "batches_per_epoch": , VALIDATION_ITERATION_SIZE
      "sampling_rates":  [1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0],
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
     "cbt_lm": {
            "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "cbt_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "schmoop_lm": {
           "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "schmoop_hierarchy": {
            "type": "basic",
            "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY
       },
       "bookscorpus_lm": {
           "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "bookscorpus_hierarchy": {
           "type": "basic",
           "batch_size": PASSAGE_BASE_BATCH_SIZE * NUM_GPUS,
           "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "filmcorpus_lm": {
           "type": "basic",
            "batch_size": LM_BASE_BATCH_SIZE * NUM_GPUS,
            "max_instances_in_memory": MAX_INSTANCES_IN_MEMORY,
       },
       "filmcorpus_hierarchy": {
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
        "atomic_lm": dataset_root + "/atomic/v4_atomic_trn.csv",
        "swag_know_lm": dataset_root + "/swagaf/data/train_full.csv",
        "cbt_lm": dataset_root + "/CBTest/data/cbt_train.txt",
        "cbt_hierarchy": dataset_root + "/CBTest/data/cbt_train.txt",
        "schmoop_lm": dataset_root + "/schmoop/stories//*//*",
        "schmoop_hierarchy": dataset_root + "/schmoop/stories//*//*",
        "bookscorpus_lm": dataset_root + "/BooksCorpus/*",
        "bookscorpus_hierarchy": dataset_root + "/BooksCorpus/*",
        "filmcorpus_lm": dataset_root + "/filmcorpus/imsdb_raw_nov_2015/imsdb_raw_nov_2015/*",
        "filmcorpus_hierarchy": dataset_root + "/filmcorpus/imsdb_raw_nov_2015/imsdb_raw_nov_2015/*",
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
        "atomic_lm": dataset_root + "/atomic/v4_atomic_dev.csv",
        "swag_know_lm": dataset_root + "/swagaf/data/val_full.csv",
        "cbt_lm": dataset_root + "/CBTest/data/cbt_valid.txt",
        "cbt_hierarchy": dataset_root + "/CBTest/data/cbt_valid.txt",
        "schmoop_lm": dataset_root + "/schmoop/stories//*//*",
        "schmoop_hierarchy": dataset_root + "/schmoop/stories//*//*",
        "bookscorpus_lm": dataset_root + "/BooksCorpus/*",
        "bookscorpus_hierarchy": dataset_root + "/BooksCorpus/*",
        "filmcorpus_lm": dataset_root + "/filmcorpus/imsdb_raw_nov_2015/imsdb_raw_nov_2015/*",
        "filmcorpus_hierarchy": dataset_root + "/filmcorpus/imsdb_raw_nov_2015/imsdb_raw_nov_2015/*",
  },
  "model": {
    "type": "know_stories",
    "lm_name": "gpt2-medium",
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
        "cbt_lm": {},
        "cbt_hierarchy": {},
        "multifile_lm": {},
        "multifile_hierarchy": {},
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
      "num_layers": 6,
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
    "num_epochs": EPOCHS,
    "validation_metric": "-loss",
    "patience": 3,
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