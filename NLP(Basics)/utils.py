"""
Utils for Amazon fine food reviews
"""
import nltk
import gensim
import numpy as np
import pandas as pd
import logging as log
import text_preprocessing
import gensim.downloader as api

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

log.basicConfig(format='%(levelname)s: %(message)s', level=log.INFO)
stop_words = (set(stopwords.words('english'))).remove('not')
sno = SnowballStemmer('english')


class nlp_on_amzn_fine_food_reviews(object):

    def get_required_data(self):
        """
        Reading Amazon fine food data and removing scores equal to 0.
        :return: dataframe
        """
        log.info(f"Started - Reading data")
        data = pd.read_csv(r"C:\Users\Siddhesh\Documents\Applied_AI\Module_3\NLP\Reviews.csv")
        data = data[data['Score'] != 3].reset_index(drop=True)
        log.info(f"Completed - (rows, columns):{data.shape}")
        return data

    def get_cleaned_data(self, df):
        """
        Remove duplicate records based on User ID, Profile name, Time, Text and Helpfulness indicator
        :param df: raw data
        :return: cleaned data
        """
        log.info(f"Started - Cleaning data")
        data = df.drop_duplicates(subset=['UserId', 'ProfileName', 'Time', 'Text'], keep='first')
        data = data[data['HelpfulnessNumerator'] <= data['HelpfulnessDenominator']]
        data = data.reset_index(drop=True)
        log.info(f"Completed - (rows, columns):{data.shape}")
        return data

    def get_preprocessed_data(self, data):
        """
        Function to get processed data cleaning HTML, punctuations, stop words
        :param data:
        :return:
        """
        log.info(f"Started - Text preprocessing")
        log.info(f"        - Removing HTML tags and Punctuations")
        log.info(f"        - Applying stemming")
        i = 0
        final_string = []
        all_positive_words = []
        all_negative_words = []

        sentences = data['Text'].values

        for sentence in sentences:
            filtered_sentence = []
            sentence = text_preprocessing.remove_html_tags(sentence)

            for word in sentence.split():
                for clean_word in text_preprocessing.remove_punctuations(word).split():
                    if clean_word.isalpha() and len(clean_word) > 2:
                        if clean_word.lower() not in stop_words:
                            s = (sno.stem(clean_word.lower())).encode('utf8')
                            filtered_sentence.append(s)

                            if data.loc[i, 'Score'] == 4 or data.loc[i, 'Score'] == 5:
                                all_positive_words.append(s)
                            elif data.loc[i, 'Score'] == 1 or data.loc[i, 'Score'] == 2:
                                all_negative_words.append(s)
                        else:
                            continue
                    else:
                        continue

            str1 = b" ".join(filtered_sentence)
            final_string.append(str1)
            i += 1

        data['CleanedText'] = final_string
        log.info(f"Completed - (rows, columns):{data.shape}")
        return data, all_negative_words, all_positive_words

    def get_freq_occurring_words(self, all_negative_words, all_positive_words, n):
        """
        Get frequently occurring words
        :param all_negative_words: list
        :param all_positive_words: list
        :return:
        """
        log.info(f"Started - Computing top {n} negative and positive words")
        freq_dist_neg = nltk.FreqDist(all_negative_words)
        freq_dist_pos = nltk.FreqDist(all_positive_words)
        log.info(f"Completed - Computing top 25 words")
        return freq_dist_neg.most_common(n), freq_dist_pos.most_common(n)

    def apply_n_gram(self, data, min, max):
        """
        Apply n-gram technique
        :param data: data frame
        :param min:
        :param max:
        :return:
        """
        count_vec = CountVectorizer(ngram_range=(min, max))
        final_count = count_vec.fit_transform(data['Text'].values)
        return final_count

    def get_freq_dist_for_ngram(self, string, min, max):
        """
        Returns frequency distribution for each sentence for n grams
        :param string:
        :param min:
        :param max:
        :return:
        """
        v = CountVectorizer(ngram_range=(min, max))
        freq_dist = v.fit([string]).vocabulary_
        return freq_dist

    def apply_tfidf(self, data, min, max):
        """
        Apply tfidf technique
        :param data: data frame
        :param min:
        :param max:
        :return:
        """
        tfidf_vec = TfidfVectorizer(ngram_range=(min, max))
        tfidf = tfidf_vec.fit_transform(data['Text'].values)
        features = tfidf_vec.get_feature_names()
        return tfidf, features

    def get_top_tfidf_features(self, row, features, top_n):
        """
        Get top n tfidf values in a row
        :param row:
        :param features:
        :param top_n:
        :return:
        """
        topn_ids = np.argsort(row)[::-1][:top_n]
        top_features = [(features[i], row[i]) for i in topn_ids]
        df = pd.DataFrame(top_features)
        df.columns = ['Features', 'TF-IDF']
        return df

    def get_similarity(self, str1, str2):
        """
        Get similarity between 2 words
        :param str1:
        :param str2:
        :return:
        """
        wv = api.load('word2vec-google-news-300')  # Takes time
        return wv.similarity(str1, str2)

    def get_similar_words(self, str1):
        """
        Get similar words
        :param str1:
        :return:
        """
        wv = api.load('word2vec-google-news-300')  # Takes time
        return wv.most_similar(str1)

    def custom_W2V_model(self, list_of_sentences, str1):
        """
        Simple W2V custom built model
        :param list_of_sentences:
        :param str1:
        :return:
        """
        wv = api.load('word2vec-google-news-300')  # Takes time
        word_list = []
        for sentence in list_of_sentences:
            filtered_sentence = []
            for word in sentence.split():
                filtered_sentence.append(word.lower())
            word_list.append(filtered_sentence)

        model = gensim.models.Word2Vec(word_list, min_count=1)

        try:
            similar_words = model.wv.most_similar(str1)
            return similar_words
        except KeyError:
            return f'{str1} not in vocabulary'




