from collections import defaultdict
import random

import pandas as pd

## INSPIRATION ##

# Build off code in this article
# http://www.onthelambda.com/2014/02/20/how-to-fake-a-sophisticated-knowledge-of-wine-with-markov-chains/

def generate_trigrams(words):
    if len(words) < 3:
        return
    for i in range(len(words) - 2):
        yield words[i], words[i + 1], words[i + 2]


class MarkovTweetGenerator:

    def __init__(self):

        self.tweets = pd.read_csv("tweets.csv", parse_dates=['created_at'])

        self.tweets["filtered_text"] = self.tweets.text.apply(self.filter_tweet)

        self.default_trigram = ("BEGIN", "NOW", "END")
        self.tweets["trigrams"] = self.tweets.filtered_text.apply(generate_trigrams)

        # keys are bigrams
        self.chain = defaultdict(list)
        self.generate_markov_chain()

    def filter_tweet(self, tweet):
        # initial text processing
        flat_tweet = tweet

        # filtering
        words = ["BEGIN", "NOW"]
        for w in flat_tweet.split():
            if w.startswith("http"):
                continue
            elif "http" in w:
                words.append(w[:w.index("http")])
            elif w in ["RT"]:
                continue
            else:
                words.append(w)
        words.append("END")

        # exclude tweets that do not contain a full trigram
        if len(words) < 6:
            words = list(self.default_trigram)
        return words

    def generate_markov_chain(self):
        # reset chain on each call
        self.chain = defaultdict(list)

        # Loop through text in sources and generate trigrams
        for gen in self.tweets.trigrams:
            # Generate trigrams and create dictionary to compute probs
            for trigram in gen:
                if trigram == self.default_trigram:
                    continue
                else:
                    self.chain[trigram[:2]].append(trigram[2])

    def generate_prediction(self):
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

