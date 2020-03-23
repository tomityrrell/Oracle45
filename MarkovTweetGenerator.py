from collections import defaultdict
import random

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


## INSPIRATION ##

# Build off code in this article
# http://www.onthelambda.com/2014/02/20/how-to-fake-a-sophisticated-knowledge-of-wine-with-markov-chains/

def generate_trigrams(words):
    if len(words) < 3:
        return []
    else:
        return list(zip(words[:-2], words[1:-1], words[2:]))


class MarkovTweetGenerator:

    def __init__(self):

        self.tweets = pd.read_csv("tweets.csv", parse_dates=['created_at'])

        self.default_trigram = ("BEGIN", "NOW", "END")
        self.tweets["filtered_text"] = self.tweets.text.apply(self.filter_tweet)
        self.tweets["trigrams"] = self.tweets["filtered_text"].apply(generate_trigrams)

        # keys are bigrams
        self.chain = self.generate_markov_chain()

        self.vectorizer = TfidfVectorizer()
        self.tdMatrix = self.vectorizer.fit_transform(self.tweets.filtered_text.apply(lambda l: ' '.join(l)))
        self.tdDataFrame = pd.SparseDataFrame(self.tdMatrix)

    def filter_tweet(self, tweet):
        # initial text processing
        flat_tweet = tweet
        if flat_tweet.startswith("RT"):
            flat_tweet = flat_tweet[flat_tweet.index(":")+1:]

        # filtering
        words = ["BEGIN", "NOW"]
        for w in flat_tweet.split():
            if w.startswith("http"):
                continue
            elif "http" in w:
                words.append(w[:w.index("http")])
            else:
                words.append(w)
        words.append("END")

        # exclude tweets that do not contain a full trigram
        if len(words) < 6:
            words = list(self.default_trigram)
        return words

    def generate_markov_chain(self):
        # reset chain on each call
        chain = defaultdict(list)

        # Loop through text in sources and generate trigrams
        for gen in self.tweets.trigrams:
            # Generate trigrams and create dictionary to compute probs
            for trigram in gen:
                if trigram == self.default_trigram:
                    continue
                else:
                    chain[trigram[:2]].append(trigram[2])

        return chain

    def generate_tweet(self):
        prediction = []

        # Random/Generic start point
        sword1 = "BEGIN"
        sword2 = "NOW"

        #     # Choose starting tweet containing seed
        #     tweet_seed = random.choice(filter(lambda s:  seed in s, sources))
        #
        #     # Take starting words from tweet_seed
        #     sword1 = tweet_seed = [2]
        #     sword2 = tweet_seed = [3]

        while True:
            sword1, sword2 = sword2, random.choice(self.chain[(sword1, sword2)])
            if sword2 == "END":
                break
            prediction.append(sword2)

        return ' '.join(prediction)

