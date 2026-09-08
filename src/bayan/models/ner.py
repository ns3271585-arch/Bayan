"""Lab 3B: NER label alignment helpers."""


def align_labels(word_ids, word_labels):
    """Align word-level BIO labels to tokenizer subwords.

    Only the first subword of each original word keeps its label.
    Special tokens and continuation subwords are masked with -100.
    """

    aligned = []
    previous_word_id = None

    for word_id in word_ids:
        # Special tokens such as CLS / SEP
        if word_id is None:
            aligned.append(-100)

        # First subword of a new word
        elif word_id != previous_word_id:
            aligned.append(word_labels[word_id])

        # Additional subword piece of the same word
        else:
            aligned.append(-100)

        previous_word_id = word_id

    return aligned