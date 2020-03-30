import re
import os
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
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

# Download the 'stopwords' and 'punkt' from the Natural Language Toolkit, you can comment the next lines if already present.
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('tagsets')
# nltk.download('averaged_perceptron_tagger')


porter = PorterStemmer()
stop_words = set(stopwords.words('english'))
word2vec_model = KeyedVectors.load_word2vec_format("word2vec_twitter_tokens.bin", binary=True, unicode_errors="ignore")

retweet_prefix = "RT "
user_name_string = "USERNAME"
hashtag_string = "HASHTAG"
link_string = "LINK"

user_name_re = "@\w{1,15}"
link_re = "(?:http:|https:)?//t.co/\w*"
hashtag_re = "#\w*"
repeated_punkt_re = "(?:\?|!){2,}"


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
    return re.sub(user_name_re, user_name_string, string)


def replace_hashtag(string):
    return re.sub(hashtag_re, hashtag_string, string)


def replace_link(string):
    return re.sub(link_re, link_string, string)


def preprocessed_tweet(text):
    if isRetweet(text):
        text = text[len(retweet_prefix)-1:]

    #text = replace_username(text)
    #text = replace_hashtag(text)
    #text = replace_link(text)

    return text


def bow_preprocessed_tweet(text, stemming=True):
    if isRetweet(text):
        text = text[len(retweet_prefix)-1:]

    text = "".join([char.lower() for char in text if not is_punctuation(char)])
    text = re.sub('\s+', ' ', text).strip()
    text = replace_username(text)
    text = replace_hashtag(text)
    text = replace_link(text)

    if stemming:
        text = stemmed_text(text)

    return text


def stemmed_text(text):
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokens = tokenizer.tokenize(text)
    stem_sentence = []
    for word in tokens:
        stem_sentence.append(porter.stem(word))
    return " ".join(stem_sentence)


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
    # cleaned_tokens = [x for x in tokens if x not in stop_words]
    text = " ".join(tokens)

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


def extract_averaged_embeddings(tokens, word2vec_model):
    vectors = []
    for token in tokens:
        try:
            vector = word2vec_model[token]
            vectors.append(vector)
        except KeyError:
            pass
            #print("The word {} does not appear in this model".format(token))

    dict = {}
    if len(vectors) > 0:
        averaged_vector = np.average(vectors, axis=0)
        for index, item in np.ndenumerate(averaged_vector):
            # indexes are represented as tuples (x,), extract x
            key = "vec"+str(index[0])
            dict[key] = item

    return dict


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


def extract_hashtags(text):
    hashtags = re.findall('#\w*', text)
    return dict(Counter(hashtags))


def punctuation_marks_count(text):
    punctuation_marks = ",.?!:;'\""

    features = {}
    for punctuation_mark in punctuation_marks:
        features[punctuation_mark] = text.count(punctuation_mark)

    return features


def number_of_capital_letters(text):
    return


def number_of_repeated_punctuation(text):
    return sum(1 for char in text if char.isupper())


def extract_features(tweet):
    # For some features text without pre-processing is used, e.g. number of punctuation marks
    # Tokens are extracted from a pre-processed tweet
    preprocessed_text = bow_preprocessed_tweet(tweet.text)
    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    tokens = tokenizer.tokenize(preprocessed_text)

    number_of_user_names = len(re.findall(user_name_re, tweet.text))
    number_of_links = len(re.findall(link_re, tweet.text))
    number_of_capital_letters = sum(1 for char in tweet.text if char.isupper())
    number_of_repeated_punctuation = len(re.findall(repeated_punkt_re, tweet.text))

    features = {"number_of_user_names": number_of_user_names,
                "number_of_links": number_of_links,
                "tweet_len": len(tweet.text),
                "adjectives_frequency": adjectives_frequency(tokens),
                "verbs_frequency": verbs_frequency(tokens),
                "capital_letters_number": number_of_capital_letters,
                "repeated_punctuation_number": number_of_repeated_punctuation}
    features.update(punctuation_marks_count(tweet.text))
    # hashtags occurences
    features.update(extract_hashtags(tweet.text))
    features.update(extract_averaged_embeddings(tokens, word2vec_model))

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


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)


# Check if K means with 5 centroids produce similar clusters
def save_clustering_results(clusters, folder):
    create_dir(folder)
    file_name = "Cluster {}.txt"

    for index, cluster in enumerate(clusters):
        cluster_file_name = file_name.format(index)
        cluster_file_path = os.path.join(folder, cluster_file_name)
        with open(cluster_file_path, 'w') as file:
            file.write("Number of tweets {} \n".format(len(cluster)))
            file.write("Dominance: {} \n".format(len([x for x in cluster if x.category == MisogynousCategory.dominance])))
            file.write("Sexual Harassment: {} \n".format(len([x for x in cluster if x.category == MisogynousCategory.sexual_harassment])))
            file.write("Derailing: {} \n".format(len([x for x in cluster if x.category == MisogynousCategory.derailing])))
            file.write("Discredit: {} \n".format(len([x for x in cluster if x.category == MisogynousCategory.discredit])))
            file.write("Stereotype: {} \n".format(len([x for x in cluster if x.category == MisogynousCategory.stereotype])))
            for tweet in cluster:
                file.write(tweet.text + "\t" + tweet.category.name + "\n")


# To create word clouds
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
    file = "results word occurences/" + file_name +".txt"
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

    individual_features = list(map(extract_features, misogyny_tweets))

    tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
    word_ngrams = [extract_ngrams(tokenizer.tokenize(bow_preprocessed_tweet(tweet.text)), n_list=[1, 2, 3])
                   for tweet in misogyny_tweets]

    char_ngrams = [extract_char_ngrams(tokenizer.tokenize(bow_preprocessed_tweet(tweet.text, stemming=False)), n_list=[3, 4, 5])
                   for tweet in misogyny_tweets]

    # do not use POS tags because they shown low performance for short texts like tweets
    #pos_tags_ngrams = [extract_pos_ngrams(tokenizer.tokenize(bow_preprocessed_tweet(tweet.text)), n_list=[1, 2, 3])
     #              for tweet in misogyny_tweets]

    hasher = FeatureHasher()
    ngrams = [];
    for index, dict1 in enumerate(individual_features):
        curr_dict = dict1.copy()
        d1 = word_ngrams[index]
        curr_dict.update(d1)
        d2 = char_ngrams[index]
        curr_dict.update(d2)
        #d3 = pos_tags_ngrams[index]
        #curr_dict.update(d3)
        ngrams.append(curr_dict)

    X_features = hasher.fit_transform(ngrams)

    #print(hasher.get_feature_names()[1:100])

    clusters = perform_k_means_clustering(X_features, misogyny_tweets)
    save_clustering_results(clusters, "results")

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