
# These are the special token ids for relationships and that have semantic meaning.

token_tags = []

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

# Special character to
token_tags += ["zBlank"]