# These are the special token ids for relationships and that have semantic meaning.

token_tags = []

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

token_tags.extend(atomic_categories)

# Use the Atomic labelling format but just add one generic tag.
swag_categories = []
swag_categories += ["oxNext"]

token_tags.extend(swag_categories)
