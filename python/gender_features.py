"""Character-bigram bag-of-ngrams features for name -> gender classification.
A classic, well-understood technique (conceptually the same family as
fastText's bag-of-ngrams) chosen specifically so the resulting classifier is
architecturally IDENTICAL to exp1's TinyMLP (Linear -> ReLU -> Linear ->
Softmax) — meaning native/mlp.cpp needs zero changes to run this real model,
since it already reads in_dim/hidden_dim/out_dim from shapes.txt at runtime.

Padding the name with ^/$ boundary markers lets bigrams capture
start-of-name and end-of-name patterns (e.g. names ending "a" skew female in
many Indian naming traditions) as well as interior character patterns.
"""


def normalize(name):
    return "^" + name.strip().lower() + "$"


def bigrams(name):
    padded = normalize(name)
    return [padded[i:i + 2] for i in range(len(padded) - 1)]


def build_vocab(names):
    vocab = sorted({bg for name in names for bg in bigrams(name)})
    return vocab


def extract_features(name, vocab):
    index = {bg: i for i, bg in enumerate(vocab)}
    vec = [0.0] * len(vocab)
    for bg in bigrams(name):
        i = index.get(bg)
        if i is not None:
            vec[i] += 1.0
    return vec
