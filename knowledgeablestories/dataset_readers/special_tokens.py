# These are the special token ids for relationships and that have semantic meaning.

token_tags = []
token_tags += ["<|endofsentence|>"]

# Special character to model blank knowledgebase tokens.
token_tags += ["zBlank"]

# These are all the atomic dataset values.

atomic_categories = []
atomic_categories += ["oEffect"]
atomic_categories += ["oReact"]
atomic_categories += ["oWant"]
atomic_categories += ["xAttr"]
atomic_categories += ["xEffect"]
atomic_categories += ["xIntent"]
atomic_categories += ["xNeed"]
atomic_categories += ["xReact"]
atomic_categories += ["xWant"]

atomic_dict = {"oEffect": 0, "oReact": 1, "oWant": 2,
               "xAttr": 3, "xEffect": 4, "xIntent": 5,
               "xNeed": 6, "xReact": 7, "xWant": 8,
               }

# token_tags.extend(atomic_categories)

# Use the Atomic labelling format but just add one generic tag.
swag_categories = []
swag_categories += ["oxNext"]

snli_dict = {"entailment": 0, "contradiction": 1, "neutral": 2}

# token_tags.extend(swag_categories)
