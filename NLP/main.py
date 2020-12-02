"""
Main script

Context
This dataset consists of reviews of fine foods from amazon. The data span a period of more than 10 years, including all
~500,000 reviews up to October 2012. Reviews include product and user information, ratings, and a plain text review. It 
also includes reviews from all other Amazon categories.
"""

from AAIC.NLP import utils

methods = utils.nlp_on_amzn_fine_food_reviews()

# Get required data
data = methods.get_required_data()

# Get cleaned data
cleaned_data = methods.get_cleaned_data(data)

# Get preprocessed data
preprocessed_data, all_negative_words, all_positive_words = methods.get_preprocessed_data(cleaned_data)
negative_words, positive_words = methods.get_freq_occurring_words(all_negative_words, all_positive_words, 25)

# Apply n-gram on whole data
n_gram_data = methods.apply_n_gram(cleaned_data, 2)

# Apply n_gram on a single string
freq_count_dict = methods.get_freq_dist_for_ngram("This product is awesome", 1, 3)

# Apply TF-IDF vectorizer
tfidf_matrix, features = methods.apply_tfidf(cleaned_data, 1, 2)
top_tfidf = methods.get_top_tfidf_features(tfidf_matrix[0, :].toarray()[0], features, 25)

# Word2Vector
list_of_sentences = ['This food is delicious', 'Very tasty', 'Excellent food', 'Yummy food']
str1 = 'tasty'
similar_words = methods.custom_W2V_model(list_of_sentences, str1)
print(similar_words)










