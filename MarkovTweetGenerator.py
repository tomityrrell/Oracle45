from collections import defaultdict
import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# INSPIRATION
# Built off code in this article:
# http://www.onthelambda.com/2014/02/20/how-to-fake-a-sophisticated-knowledge-of-wine-with-markov-chains/

default_head = ("BEGINN", "NNOW")
default_tail = ("ENND", "NNOW")


def filter_tweet(tweet):
    # initial text processing
    flat_tweet = tweet
    if flat_tweet.startswith("RT"):
        flat_tweet = flat_tweet[flat_tweet.index(":")+1:]

    # filtering
    words = list(default_head)
    for w in flat_tweet.split():
        if w.startswith("http"):
            continue
        elif "http" in w:
            words.append(w[:w.index("http")])
        else:
            words.append(w)
    words += list(default_tail)

    # exclude tweets that do not contain a full trigram
    if len(words) < 6:
        words = []
    return words


def generate_trigrams(words):
    if len(words) < 3:
        return []
    else:
        return list(zip(words[:-2], words[1:-1], words[2:]))


def generate_markov_chain(trigrams):

    chain = defaultdict(list)

    # Generate trigrams and create dictionary to compute probs
    for trigram in trigrams:
        chain[trigram[:2]].append(trigram[1:])

    return chain


def generate_tweet(chain, head=default_head, tail=default_tail):

    prediction = []

    current_bigram = random.choice(chain[head])
    while current_bigram[1] != tail[0]:
        prediction.append(current_bigram[1])
        current_bigram = random.choice(chain[current_bigram])

    return ' '.join(prediction)


def generate_from(seed, chain):
    seed_bigrams = []
    for bigram in chain:
        if seed in bigram[0] or seed in bigram[1]:
            seed_bigrams.append(bigram)
    seed_bigram = random.choice(seed_bigrams)

    prediction = []

    forward_seed = generate_tweet(forward_chain, head=seed_bigram).split()
    reverse_seed = generate_tweet(reverse_chain, head=seed_bigram[::-1], tail=default_head[::-1]).split()[::-1]

    prediction = reverse_seed + list(seed_bigram) + forward_seed

    return " ".join(prediction)


tweets = pd.read_csv("tweets.csv", parse_dates=['created_at'])

tweets["filtered_text"] = tweets.text.apply(filter_tweet)
tweets["trigrams"] = tweets["filtered_text"].apply(generate_trigrams)
forward_chain = generate_markov_chain(tweets["trigrams"].sum())
reverse_chain = generate_markov_chain(tweets["filtered_text"].apply(lambda l: l[::-1]).apply(generate_trigrams).sum())

vectorizer = TfidfVectorizer(analyzer='word')
tfMatrix = vectorizer.fit_transform(tweets.filtered_text.apply(lambda l: ' '.join(l)))
tfDataFrame = pd.SparseDataFrame(tfMatrix)
