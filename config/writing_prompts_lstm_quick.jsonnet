local dataset_root = std.extVar("DATASET_ROOT");
local dataset_cache_root = std.extVar("DATASET_CACHE_ROOT");
local embedder_vocab_size = std.parseInt(std.extVar("EMBEDDER_VOCAB_SIZE"));
local NUM_GPUS = std.parseInt(std.extVar("NUM_GPUS"));
local NUM_CPUS = std.parseInt(std.extVar("NUM_CPUS"));
local NUM_READER_CPUS = NUM_CPUS - 2;
local NUM_ITERATOR_CPUS = NUM_CPUS - 2;
local WP_BASE_BATCH_SIZE = 2;

{
  "dataset_reader": {
    "type": "multitask_reader",
    "datasets_for_vocab_creation": [],
    "dataset_readers": {
              "writing_prompts_lm": {
                "type": "multiprocess_unreleased",
                "base_reader": {
                    "type": "writing_prompts_lm",
                },
                "num_workers": NUM_READER_CPUS,
            },
            "writing_prompts_hierarchy": {
                "type": "multiprocess_unreleased",
                "base_reader": {
                   "type": "writing_prompts_hierarchy",
                },
                "num_workers": NUM_READER_CPUS,
            }
        },
  },
  "iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy"],
   "iterate_forever": false,
   "instances_per_epoch": 5000,
   "iterators": {
      "writing_prompts_lm": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": WP_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": NUM_ITERATOR_CPUS,
       },
       "writing_prompts_hierarchy": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": WP_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": NUM_ITERATOR_CPUS,
       },
    },
  },
  "validation_iterator": {
   "type": "multitask_iterator",
   "names_to_index": ["writing_prompts_lm", "writing_prompts_hierarchy"],
   "iterate_forever": false,
   "instances_per_epoch": 500,
   "iterators": {
        "writing_prompts_lm": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": WP_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": NUM_ITERATOR_CPUS,
       },
       "writing_prompts_hierarchy": {
           "type": "multiprocess_unreleased",
            "base_iterator": {
                "type": "basic",
                "batch_size": WP_BASE_BATCH_SIZE * NUM_GPUS,
            },
            "num_workers": NUM_ITERATOR_CPUS,
       },
    },
  },
  "train_data_path": {
        "writing_prompts_lm": dataset_root + "/WritingPrompts/valid_split/*",
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/valid_split/*",
  },
  "validation_data_path": {
        "writing_prompts_lm": dataset_root + "/WritingPrompts/valid_split/*",
        "writing_prompts_hierarchy": dataset_root + "/WritingPrompts/valid_split/*",
  },
  "model": {
    "type": "knowledgeable_stories",
    "embedder_vocab_size": embedder_vocab_size,
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
  },
  "trainer": {
    "num_epochs": 10,
    "validation_metric": "-loss",
    "patience": 2,
    "grad_norm": 2.0,
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