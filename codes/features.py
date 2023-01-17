# -*- coding: utf-8 -*-
"""
Last modified on 16th Jan.
Author: Hung.

time spark-submit --master yarn --deploy-mode cluster  create_parquet.py

"""
import re
import nltk
from nltk.tokenize import sent_tokenize


def word_count(content):
    return len(content.split())


def clean_data(text):
    rules = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = text.lower()
    codeless_text = re.sub("<code>.*?</code>", "", text)  # remove code block
    cleantext = re.sub(rules, '', codeless_text)
    return cleantext


def syllable_count(word):
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if word.endswith('le'):
        count += 1
    if count == 0:
        count += 1
    return count


def char_count(content):
    return len(content)


def sentence_count(content):
    return len(sent_tokenize(content))


# Coleman–Liau_index
def coleman_liau(cleaned_text):
    no_chars = char_count(cleaned_text)
    no_sentences = sentence_count(cleaned_text)
    no_words = word_count(cleaned_text)
    L = (no_chars / no_words) * 100
    S = (no_sentences / no_words) * 100
    return 0.0588 * L - 0.296 * S - 15.8


# Flesch-Reading ease
def flesch_ease(cleaned_text):
    no_syllables = syllable_count(cleaned_text)
    no_words = word_count(cleaned_text)
    no_sentences = sentence_count(cleaned_text)
    words_per_sent = no_words / no_sentences
    syll_per_word = no_syllables / no_words
    return 206.835 - (1.015 * words_per_sent) - (84.6 * syll_per_word)


# Flesch-Reading ease
def flesch_grade(cleaned_text):
    no_syllables = syllable_count(cleaned_text)
    no_words = word_count(cleaned_text)
    no_sentences = sentence_count(cleaned_text)
    words_per_sent = no_words / no_sentences
    syll_per_word = no_syllables / no_words
    return 0.39 * words_per_sent + 11.8 * syll_per_word - 15.59


def code_percentage(content):
    # numbers between <code> tags / #no chars total
    
    pass
