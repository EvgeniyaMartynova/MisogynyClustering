import re
import os
import string
from typing import Set, Any

import numpy as np
import nltk
from collections import Counter
import sklearn.metrics
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
from emoji import UNICODE_EMOJI
from gensim.models import KeyedVectors
import utils

# Download the 'stopwords' and 'punkt' from the Natural Language Toolkit, you can comment the next lines if already present.
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('tagsets')
# nltk.download('averaged_perceptron_tagger')


stop_words: Set[Any] = set(stopwords.words('english'))
tokenizer = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=True)

retweet_prefix = "RT "
user_name_string = "USERNAME"
hashtag_string = "HASHTAG"
link_string = "LINK"

user_name_re = "@\w{1,15}"
link_re = "(?:http:|https:)?//t.co/\w*"
hashtag_re = "#\w*"
repeated_punkt_re = "(?:\?{2,}|!{2,}|\.{2,}|(?:\?!){2,})"
lenghtening_re = "(?:a{3,}|b{3,}|c{3,}|d{3,}|e{3,}|f{3,}|g{3,}|h{3,}|i{3,}|j{3,}|k{3,}|l{3,}|m{3,}|n{3,}|o{3,}|p{3,}|" \
                 "q{3,}|r{3,}|s{3,}|t{3,}|u{3,}|v{3,}|w{3,}|x{3,}|y{3,}|z{3,})"


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


# lazy loading of word embeddings
class Word2vec:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = KeyedVectors.load_word2vec_format("word2vec_twitter_tokens.bin", binary=True, unicode_errors="ignore")
        return self._model


def create_dir(path):
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)


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

    return tweets


def load_pos_tags():
    return load_words_from_file('pos_tags.txt')


def misogyny_only_tweets(tweets):
    misogyny_tweets = [tweet for tweet in tweets if tweet.misogynous]
    return misogyny_tweets


def tweet_by_category(tweets, category):
    cat_tweets = [tweet for tweet in tweets if tweet.category == category]
    return cat_tweets


def isRetweet(tweet):
    return tweet.startswith(retweet_prefix)


def replace_username(tweet):
    return re.sub(user_name_re, user_name_string, tweet)


def replace_hashtag(tweet):
    return re.sub(hashtag_re, hashtag_string, tweet)


def replace_link(tweet):
    return re.sub(link_re, link_string, tweet)


def is_emoji(token):
    return token in UNICODE_EMOJI


def preprocessed_tweet(text):
    if isRetweet(text):
        text = text[len(retweet_prefix)-1:]

    return text


def bow_preprocessed_tweet(text):
    if isRetweet(text):
        text = text[len(retweet_prefix)-1:]

    # lowercase
    text = "".join([char.lower() for char in text])
    text = replace_username(text)
    text = replace_hashtag(text)
    text = replace_link(text)
    # remove punctuation
    #text = "".join([char for char in text if not is_punctuation(char)])
    # remove extra spaces
    text = re.sub('\s+', ' ', text).strip()

    return text


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


def collection_vocabulary(tweets):
    vocabulary = Counter()
    for tweet in tweets:
        preprocessed_text = bow_preprocessed_tweet(tweet.text)
        tokens = tokenizer.tokenize(preprocessed_text)
        for token in tokens:
            vocabulary[token] += 1

    return vocabulary


# vocabulary is a counter object
def rare_words(vocabulary, threshold=3):
    words = []
    for key in vocabulary.keys():
        if vocabulary[key] < threshold:
            words.append(key)

    return words


def extract_ngrams(tokens, n_list, words_to_exclude):
    cleaned_counter = Counter()
    cleaned_tokens = [x for x in tokens if not is_punctuation(x)]

    for n in n_list:
        grams = list(ngrams(cleaned_tokens, n))

        for word_tuple in grams:
            for word in word_tuple:
                if word not in words_to_exclude:
                    cleaned_counter[" ".join(word_tuple)] += 1
                    break

    return dict(cleaned_counter)


def extract_char_ngrams(tokens, n_list):
    cleaned_counter = Counter()
    cleaned_tokens = [x for x in tokens if not is_punctuation(x) and x not in stop_words]
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


def extract_averaged_embeddings(tokens, word2vec):
    vectors = []
    for token in tokens:
        try:
            vector = word2vec.model[token]
            vectors.append(vector)
        except KeyError:
            pass

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
    punctuation_marks = ",.?!:;<>&'\""

    features = {}
    for punctuation_mark in punctuation_marks:
        features[punctuation_mark] = text.count(punctuation_mark)

    return features


def extract_features(tweet, word2vec=None):
    # For some features text without pre-processing is used, e.g. number of punctuation marks
    # Tokens are extracted from a pre-processed tweet
    preprocessed_text = bow_preprocessed_tweet(tweet.text)
    tokens = tokenizer.tokenize(preprocessed_text)

    number_of_repeated_punctuation = len(re.findall(repeated_punkt_re, tweet.text))
    number_of_links = len(re.findall(link_re, tweet.text))
    number_of_emojis = sum(1 for token in tokens if is_emoji(token))
    number_of_word_lengthening = len(re.findall(lenghtening_re, tweet.text))

    features = {#"number_of_links": number_of_links,
                "adjectives_frequency": adjectives_frequency(tokens),
                "verbs_frequency": verbs_frequency(tokens)}
                #"number_of_emojis": number_of_emojis,
                #"repeated_punctuation_number": number_of_repeated_punctuation,
                #"number_of_word_lengthening": number_of_word_lengthening}
    # hashtags occurences
    #features.update(extract_hashtags(tweet.text))

    #features = {}
    if word2vec is not None:
        # remove punctuation marks, emojis, username, token, hashtag
        tokens = [x for x in tokens if not is_punctuation(x) and x not in stop_words]
        features.update(extract_averaged_embeddings(tokens, word2vec))

    return features


def perform_k_means_clustering(feature_vectors, tweets, n_clusters=5):
    #feature_vectors = list(map(extract_features, tweets))
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(feature_vectors)

    clusters = []
    for cluster_index in np.unique(labels):
        current_cluster = []
        for index, value in enumerate(labels):
            if value == cluster_index:
                current_cluster.append(tweets[index])
        clusters.append(current_cluster)

    return clusters, labels


# Check if K means with 5 centroids produce similar clusters
def save_clustering_results(clusters, metrics, folder):
    matrics_file_path = os.path.join(folder, "evaluation_metrics.txt")
    with open(matrics_file_path, 'w') as file:
        file.write("Silhouette score: {} \n".format(metrics["silhouette_score"]))
        file.write("Calinski harabaz score: {} \n".format(metrics["calinski_harabaz_score"]))
        file.write("Davies bouldin score: {} \n".format(metrics["davies_bouldin_score"]))

    file_name = "Cluster {}.txt"

    for index, cluster in enumerate(clusters):
        cluster_file_name = file_name.format(index+1)
        cluster_file_path = os.path.join(folder, cluster_file_name)
        with open(cluster_file_path, 'w') as file:
            tweets_num = len(cluster)
            file.write("Number of tweets {} \n".format(tweets_num))
            dominance_num = len([x for x in cluster if x.category == MisogynousCategory.dominance])
            file.write("Dominance: {}, share {} \n".format(dominance_num, dominance_num/tweets_num))
            harassment_num = len([x for x in cluster if x.category == MisogynousCategory.sexual_harassment])
            file.write("Sexual Harassment: {}, share {} \n".format(harassment_num, harassment_num/tweets_num))
            derailing_num = len([x for x in cluster if x.category == MisogynousCategory.derailing])
            file.write("Derailing: {}, share {} \n".format(derailing_num, derailing_num/tweets_num))
            discredit_num = len([x for x in cluster if x.category == MisogynousCategory.discredit])
            file.write("Discredit: {}, share {} \n".format(discredit_num, discredit_num/tweets_num))
            stereotype_num = len([x for x in cluster if x.category == MisogynousCategory.stereotype])
            file.write("Stereotype: {}, share {} \n".format(stereotype_num, stereotype_num/tweets_num))
            for tweet in cluster:
                file.write(tweet.text + "\t" + tweet.category.name + "\n")



# word embeddings + linguistic features
def get_features(tweets, word_ngrams_list=[], char_ngrams_list=[3,4,5], use_embeddings=False, words_to_exclude=[]):
    individual_features = []
    word2vec = Word2vec() if use_embeddings else None
    for tweet in tweets:
        tweet_features = extract_features(tweet, word2vec)
        individual_features.append(tweet_features)

    word_ngrams = None
    if len(word_ngrams_list) > 0:
        word_ngrams = [extract_ngrams(tokenizer.tokenize(bow_preprocessed_tweet(tweet.text)), word_ngrams_list, words_to_exclude)
                       for tweet in tweets]

    char_ngrams = None
    if len(char_ngrams_list) > 0:
        char_ngrams = [extract_char_ngrams(tokenizer.tokenize(bow_preprocessed_tweet(tweet.text)), n_list=char_ngrams_list)
                       for tweet in tweets]

    features = []
    for index, dict in enumerate(individual_features):
        curr_dict = dict.copy()

        if word_ngrams is not None:
            word_ngrams_dict = word_ngrams[index]
            curr_dict.update(word_ngrams_dict)

        if char_ngrams is not None:
            char_ngrams_dict = char_ngrams[index]
            curr_dict.update(char_ngrams_dict)

        features.append(curr_dict)

    hasher = DictVectorizer()
    X_features = hasher.fit_transform(features)
    return X_features


# word embeddings + linguistic features
def clustering(tweets, output_folder, word_ngrams_list=[], char_ngrams_list=[3,4,5], use_embeddings=False, words_to_exclude=[]):
    features = get_features(tweets, word_ngrams_list, char_ngrams_list, use_embeddings, words_to_exclude)
    clusters, labels = perform_k_means_clustering(features, tweets)
    metrics = utils.internalValidation(features.toarray(), labels)
    create_dir(output_folder)
    save_clustering_results(clusters, metrics, output_folder)


def find_k(tweets, word_ngrams_list=[], char_ngrams_list=[3,4,5], use_embeddings=False, words_to_exclude=[]):
    features = get_features(tweets, word_ngrams_list, char_ngrams_list, use_embeddings, words_to_exclude)
    # calculate distortion for a range of number of cluster
    distortions = []
    silhouette_scores = []
    calinski_harabaz_scores = []
    davies_bouldin_scores = []
    ks = range(2, 11)
    for i in ks:
        print("K = {}".format(i))
        km = KMeans(n_clusters=i, random_state=42)
        labels = km.fit_predict(features)
        distortions.append(km.inertia_)
        metrics = utils.internalValidation(features.toarray(), labels)
        silhouette_scores.append(metrics["silhouette_score"])
        calinski_harabaz_scores.append(metrics["calinski_harabaz_score"])
        davies_bouldin_scores.append(metrics["davies_bouldin_score"])

    fig, axs = plt.subplots(2, 2)
    # plot
    axs[0, 0].plot(ks, distortions, marker='o')
    axs[0, 0].set_title('SSE')
    axs[0, 1].plot(ks, silhouette_scores, marker='^')
    axs[0, 1].set_title('Silhouette Scores')
    axs[1, 0].plot(ks, calinski_harabaz_scores, marker='<')
    axs[1, 0].set_title('Calinski Harabaz Scores')
    axs[1, 1].plot(ks, davies_bouldin_scores, marker='>')
    axs[1, 1].set_title('Davies Bouldin Scores')

    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.show()


def misogynyous_tweets_stats(tweets):
    tweets_count = len(tweets)
    print("Tweets number {}".format(tweets_count))
    for category in MisogynousCategory:
        if category is not MisogynousCategory.no:
            category_count = len(tweet_by_category(tweets, category))
            print("{} number {}, share {}".format(category.name, category_count, category_count/tweets_count))


def dataset_categories_analysis(tweets, plot=False):
    for category in MisogynousCategory:
        if category is not MisogynousCategory.no:
            category_distr = word_distr(tweet_by_category(tweets, category))
            if plot:
                plot_word_dist_as_cloud(category_distr, category.name)
            word_distr_to_file(category_distr.most_common(25), category.name)


# To create word clouds
def word_distr(category_tweets):
    word_dist = FreqDist()
    for tweet in category_tweets:
        tokens = tokenizer.tokenize(tweet.text)
        tokens = [x for x in tokens if x not in stop_words and not is_punctuation(x)]
        dist = FreqDist(tokens)
        word_dist.update(dist)
    return word_dist


def word_distr_to_file(word_dist, file_name):
    file = "results word occurences1/" + file_name +".txt"
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
    stop_words.add("u")
    stop_words.add("link")
    stop_words.add("username")
    stop_words.add("hashtag")
    stop_words.add("bitch")
    tweets = load_tweets("data/tweets.tsv")
    misogyny_tweets = misogyny_only_tweets(tweets)
    print(len(misogyny_tweets))

    #discredit = tweet_by_category(misogyny_tweets, MisogynousCategory.discredit)

    rare_words_array = rare_words(collection_vocabulary(misogyny_tweets), threshold=3)
    words_to_exclude = stop_words.union(set(rare_words_array))

    # elbow method discredit category
    find_k(misogyny_tweets,
           word_ngrams_list=[1],
           char_ngrams_list=[3,4],
           use_embeddings=True,
           words_to_exclude=words_to_exclude)

    """
    clustering(misogyny_tweets,
               "results/embeddings + unigrams + char 3,4 + adj + verbs",
               word_ngrams_list=[1],
               char_ngrams_list=[3,4],
               use_embeddings=True,
               words_to_exclude=words_to_exclude)
    """



if __name__ == '__main__':
    main()