local pwd = std.extVar("PWD");
local pwd2 = std.strReplace(pwd,'model','');
local train_path = pwd2 + "/fixtures/data/atomic_small.csv";

{
  "dataset_reader": {
    "type": "atomic"
  },
  "train_data_path": train_path ,
  "validation_data_path":  train_path,
  "model": {
    "type": "knowledgeable_stories",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": "gpt2"
        }
      }
    },
    "language_model_head": {
      "type": "gpt2",
      "model_name": "gpt2"
    },
    "relation_text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 768,
          "trainable": true,
          "vocab_namespace": "relation"
        }
      }
    }
  },
  "iterator": {
    "type": "basic",
    "batch_size": 32
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device": -1,
    "optimizer": {
      "type": "sgd",
      "lr": 0.01
    }
  }
}