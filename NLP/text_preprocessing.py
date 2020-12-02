"""
Text Preprocessing

"""

import re


def remove_html_tags(sentence):
    """
    Function to remove a sentence having HTML tags
    :param sentence: string
    :return: string
    """
    regex = re.compile(pattern='<.*?>')
    clean_text = re.sub(regex, ' ', sentence)
    return clean_text


def remove_punctuations(word):
    """
    Function to remove punctuation marks from a word
    :param word:
    :return:
    """
    cleaned_sentence = re.sub(pattern=r'[?|!|\|"|#|\']', repl=r'', string=word)
    cleaned_sentence = re.sub(pattern=r'[.|,|)|(|\|/]', repl=r'', string=cleaned_sentence)
    return cleaned_sentence
