import re
import string
import numpy as np
import nltk
from collections import Counter
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from nltk import ngrams
from nltk.corpus import stopwords
from nltk import download, FreqDist
from nltk.tokenize import TweetTokenizer
from enum import Enum
from nltk.probability import MLEProbDist
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer

# Download the 'stopwords' and 'punkt' from the Natural Language Toolkit, you can comment the next lines if already present.
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('tagsets')
# nltk.download('averaged_perceptron_tagger')

stop_words = set(stopwords.words('english'))

retweet_prefix = "RT "
user_name_string = "USERNAME"
hashtag_string = "HASHTAG"
link_string = "LINK"

class MisogynousCategory(Enum):
    dominance = 1
    sexual_harassment = 2
    derailing = 3
    discredit = 4
    stereotype = 5
    no = 6


class Tweet:

    def __init__(self, tweetId, text, misogynous, category, activeTarget):
        self.tweetId = tweetId
        self.text = text
        self.misogynous = misogynous
        self.category = category
        self.activeTarget = activeTarget


def load_words_from_file(file_name):
    words = []
    with open(file_name) as file:
        for word in file:
            words.append(word.strip())
    return words


def load_tweets(path):
    tweets = []
    with open(path) as tweets_file:
        for line in tweets_file:
            components = line.strip().split('\t')
            preprocessed_text = preprocessed_tweet(components[1])
            misogynous = components[2] == "1"
            tweet = Tweet(components[0], preprocessed_text, misogynous, MisogynousCategory[components[3]] if misogynous else MisogynousCategory.no, components[4] == "active")
            tweets.append(tweet)

    assert len(tweets) == 5000
    return tweets


def load_pos_tags():
    return load_words_from_file('pos_tags.txt')


def misogyny_only_tweets(tweets):
    misogyny_tweets = [tweet for tweet in tweets if tweet.misogynous]
    return misogyny_tweets


def tweet_by_category(tweets, category):
    cat_tweets = [tweet for tweet in tweets if tweet.category == category]
    return cat_tweets


def isRetweet(string):
    return string.startswith(retweet_prefix)


def replace_username(string):
    return re.sub('@\w{1,15}', user_name_string, string)


def replace_hashtag(string):
    return re.sub('#\w*', hashtag_string, string)


def replace_link(string):
    return re.sub('(?:http:|https:)?//t.co/\w*', link_string, string)


def preprocessed_tweet(text):
    if isRetweet(text):
        text = text[len(retweet_prefix)-1:]

    #text = replace_username(text)
    #text = replace_hashtag(text)
    #text = replace_link(text)

    return text


def bow_preprocessed_tweet(text):
    if isRetweet(text):
        text = text[len(retweet_prefix)-1:]

    text_nopunct = "".join([char.lower() for char in text if not is_punctuation(char)])
    text_no_doublespace = re.sub('\s+', ' ', text_nopunct).strip()
    #text = replace_username(text)
    #text = replace_hashtag(text)
    #text = replace_link(text)

    return text_no_doublespace


def has_link(text):
    return link_string in text


def has_hashtag(text):
    return hashtag_string in text


def has_username(text):
    return user_name_string in text


def is_punctuation(token):
    for char in token:
        if char not in string.punctuation and char != '’' and char != '…':
            return False

    return True


def extract_ngrams(tokens, n_list):
    cleaned_counter = Counter()

    for n in n_list:
        grams = list(ngrams(tokens, n))

        for word_tuple in grams:
            for word in word_tuple:
                if word not in stop_words:
                    cleaned_counter[" ".join(word_tuple)] += 1
                    break

    return dict(cleaned_counter)


def extract_char_ngrams(tokens, n_list):
    cleaned_counter = Counter()
    cleaned_tokens = [x for x in tokens if x not in stop_words and not is_punctuation(x)]
    text = " ".join(cleaned_tokens)

    for n in n_list:
        grams = list(ngrams(text, n))
        for gram in grams:
            cleaned_counter["".join(gram)] += 1

    return dict(cleaned_counter)


def extract_pos_ngrams(tokens, n_list):
    cleaned_counter = Counter()
    tagged_tokens = nltk.pos_tag(tokens)
    pos_tags = [tag for (word, tag) in tagged_tokens]

    for n in n_list:
        grams = list(ngrams(pos_tags, n))
        for pos_tuple in grams:
            cleaned_counter[" ".join(pos_tuple)] += 1

    return dict(cleaned_counter)


def adjectives_frequency(tokens):
    tagged_tokens = nltk.pos_tag(tokens)
    adjective_tags = ["JJ", "JJR", "JJS"]

    adj_frequencies = 0
    for (word, tag) in tagged_tokens:
        if tag in adjective_tags:
            adj_frequencies += 1

    return adj_frequencies


def verbs_frequency(tokens):
    tagged_tokens = nltk.pos_tag(tokens)
    adjective_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]

    adj_frequencies = 0
    for (word, tag) in tagged_tokens:
        if tag in adjective_tags:
            adj_frequencies += 1

    return adj_frequencies


def extract_embeddings():
    return []


def extract_hashtags(text):
    hashtags = re.findall('#\w*', text)
    return dict(Counter(hashtags))


def punctuation_marks_count(text):
    punctuation_marks = ",.?!:;'\""

    features = []
    for punctuation_mark in punctuation_marks:
        features.append(text.count(punctuation_mark))

    return features


def extract_features(tweet):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokens = tokenizer.tokenize(tweet.text)
    tokens = [x for x in tokens if x not in stop_words and not is_punctuation(x)]

    number_of_user_names = len(re.findall("@\w{1,15}", tweet.text))
    number_of_links = len(re.findall("(?:http:|https:)?//t.co/\w*", tweet.text))

    features = {"number_of_user_names": number_of_user_names,
                "number_of_links": number_of_links,
                "tweet_len": len(tweet.text),
                "adjectives_frequency": adjectives_frequency(tokens),
                "verbs_frequency": verbs_frequency(tokens),
                "punctuation_marks": punctuation_marks_count(tweet.text)}
    features.update(extract_hashtags(tweet.text))

    return features


def perform_k_means_clustering(feature_vectors, tweets, n_clusters=5):
    #feature_vectors = list(map(extract_features, tweets))
    kmeans = KMeans(n_clusters=n_clusters,random_state=42).fit(feature_vectors)
    y_kmeans = kmeans.predict(feature_vectors)
    clusters = []
    for cluster_index in np.unique(y_kmeans):
        current_cluster = []
        for index, value in enumerate(y_kmeans):
            if value == cluster_index:
                current_cluster.append(tweets[index])
        clusters.append(current_cluster)

    return clusters


def word_distr(category_tweets):
    word_dist = FreqDist()
    for tweet in category_tweets:
        tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        tokens = tokenizer.tokenize(tweet.text)
        tokens = [x for x in tokens if x not in stop_words and not is_punctuation(x)]
        dist = FreqDist(tokens)
        word_dist.update(dist)
    return word_dist


def word_distr_to_file(word_dist, file_name):
    file = "results/" + file_name +".txt"
    with open(file, "w") as filehandle:
        filehandle.writelines("{} {}\n".format(word, count) for (word, count) in word_dist)


def plot_word_dist_as_cloud(word_dist, file_name=None, plot=False):
    prob_dist = MLEProbDist(word_dist)
    viz_dict = {}
    for word_tuple in word_dist:
        string = ' '.join(word_tuple)
        viz_dict[string] = prob_dist.prob(word_tuple)

    wordcloud = WordCloud(max_words=100).generate_from_frequencies(viz_dict)
    if file_name != None:
        wordcloud.to_file("img/" + file_name +".png")

    if plot:
        plt.figure()
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


def main():
    tweets = load_tweets("data/tweets.tsv")
    misogyny_tweets = misogyny_only_tweets(tweets)
    stop_words.add("u")
    print(len(misogyny_tweets))

    hasher = FeatureHasher()
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    individual_features = list(map(extract_features, misogyny_tweets))

    word_ngrams = [extract_ngrams(tokenizer.tokenize(bow_preprocessed_tweet(tweet.text)), n_list=[1, 2, 3])
                   for tweet in misogyny_tweets]

    char_ngrams = [extract_char_ngrams(tokenizer.tokenize(bow_preprocessed_tweet(tweet.text)), n_list=[3, 4, 5])
                   for tweet in misogyny_tweets]

    pos_tags_ngrams = [extract_pos_ngrams(tokenizer.tokenize(bow_preprocessed_tweet(tweet.text)), n_list=[1, 2, 3])
                   for tweet in misogyny_tweets]


    ngrams = [];
    for index, dict1 in enumerate(individual_features):
        curr_dict = dict1.copy()
        d1 = word_ngrams[index]
        curr_dict.update(d1)
        d2 = char_ngrams[index]
        curr_dict.update(d2)
        d3 = pos_tags_ngrams[index]
        curr_dict.update(d3)
        ngrams.append(curr_dict)

    X_features = hasher.fit_transform(ngrams)

    #trigram_vectorizer = CountVectorizer(ngram_range=(1, 3), token_pattern=r'\b\w+\b', min_df=1)
    #X_2 = trigram_vectorizer.fit_transform([x.text for x in misogyny_tweets]).toarray()

    #print(hasher.get_feature_names()[1:100])

    labels = perform_k_means_clustering(X_features, misogyny_tweets)


    # categories analysis
    if False:
        # dominance
        dominance_distr = word_distr(tweet_by_category(tweets, MisogynousCategory.dominance))
        plot_word_dist_as_cloud(dominance_distr, MisogynousCategory.dominance.name)
        word_distr_to_file(dominance_distr.most_common(25), MisogynousCategory.dominance.name)
        # sexual harassment
        sexual_harassment_distr = word_distr(tweet_by_category(tweets, MisogynousCategory.sexual_harassment))
        plot_word_dist_as_cloud(sexual_harassment_distr, MisogynousCategory.sexual_harassment.name)
        word_distr_to_file(sexual_harassment_distr.most_common(25), MisogynousCategory.sexual_harassment.name)
        # derailing
        derailing_distr = word_distr(tweet_by_category(tweets, MisogynousCategory.derailing))
        plot_word_dist_as_cloud(derailing_distr, MisogynousCategory.derailing.name)
        word_distr_to_file(derailing_distr.most_common(25), MisogynousCategory.derailing.name)
        # discredit
        discredit_distr = word_distr(tweet_by_category(tweets, MisogynousCategory.discredit))
        plot_word_dist_as_cloud(discredit_distr, MisogynousCategory.discredit.name)
        word_distr_to_file(discredit_distr.most_common(25), MisogynousCategory.discredit.name)
        # stereotype
        stereotype_distr = word_distr(tweet_by_category(tweets, MisogynousCategory.stereotype))
        plot_word_dist_as_cloud(stereotype_distr, MisogynousCategory.stereotype.name)
        word_distr_to_file(stereotype_distr.most_common(25), MisogynousCategory.stereotype.name)





if __name__ == '__main__':
    main()