# -*- coding: utf-8 -*-
from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from bs4 import BeautifulSoup
#from features import Features
from pyspark.sql.functions import udf, col
# -*- coding: utf-8 -*-
"""
Last modified on 16th Jan.
Author: Hung.

time spark-submit --master yarn --deploy-mode cluster  create_parquet.py

"""
import re
import nltk
from nltk.tokenize import sent_tokenize


def clean_data(content):
    rules = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = content.lower()
    codeless_text = re.sub("<code>.*?</code>", "", text)  # remove code block
    cleantext = re.sub(rules, '', codeless_text)
    return cleantext


class Stats:
    def __init__(self, content):
        self.content = content



    def word_count(self):
        return len(self.content.split())

    def syllable_count(self):
        count = 0
        vowels = "aeiouy"
        word = self.content
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

    def char_count(self):
        return len(self.content)

    def sentence_count(self):
        return len(sent_tokenize(self.content))


class Features:
    def __init__(self, content):
        self.text = content
        self.cleaned_text = clean_data(self.text)
        stats = Stats(self.cleaned_text)

        self.no_chars = stats.char_count()
        self.no_sentences = stats.sentence_count()
        self.no_words = stats.word_count()
        self.no_syllables = stats.syllable_count()

    def get_content(self):
        return self.cleaned_text

    # Coleman–Liau_index
    def coleman_liau(self):
        L = (self.no_chars / self.no_words) * 100
        S = (self.no_sentences / self.no_words) * 100
        return 0.0588 * L - 0.296 * S - 15.8

    # Flesch-Reading ease
    def flesch_ease(self):
        words_per_sent = self.no_words / self.no_sentences
        syll_per_word = self.no_syllables / self.no_words
        return 206.835 - (1.015 * words_per_sent) - (84.6 * syll_per_word)

    # Flesch-Reading ease
    def flesch_grade(self):
        words_per_sent = self.no_words / self.no_sentences
        syll_per_word = self.no_syllables / self.no_words
        return 0.39 * words_per_sent + 11.8 * syll_per_word - 15.59

    def code_percentage(self):
        # numbers between <code> tags / #no chars total
        code = None
        # Pass self.text as the uncleaned date
        for i in re.findall("<code>(.*?)</code>", self.text):
            code += i
        no_chars_code = len(code)
        return (no_chars_code / self.no_chars) * 100




# Factors influencing how fast your questions getting answer (AnswerCount, CommentCount, ClosedDate…)? Including (long) code in the question?
# -Sentiment and scope of the title
# -Tag selection (number of tags/questions, synonyms)

# time spark-submit --master yarn --deploy-mode cluster --conf  spark.dynamicAllocation.maxExecutors=10 music.py
BIG_DATA_PATH = "/user/s2812940/project/parquet_data/posts.parquet"
DATA_PATH = "/user/s2812940/project/parquet_data/posts.parquet/part-00000-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet"
spark = SparkSession.builder.appName("Colyn").getOrCreate()

posts_df = spark.read.parquet(DATA_PATH)
# Returns questions with an accepted answer
questions = posts_df.na.drop(subset=["_AcceptedAnswerId"])
# PostTypeId not relevant anymore since this is filters
ignore_cols_posts = ("_LastEditDate", "_OwnerUserId",
                     "_LastEditorUserId", "_LastEditorDisplayName",
                     "_LastEditDate", "_LastActivityDate", "_ContentLicense",
                     "_CommunityOwnedDate")

questions = questions.drop(*ignore_cols_posts)


def flesche_ease(content):
    features = Features(content)
    return features.flesch_ease()


flesche_ease_udf = udf(lambda x: flesche_ease(x))

# Initialize a copy to do operations on and save later
metrics = questions.select(col("_AcceptedAnswerId", col("_Body")))

metrics = metrics.withColumn('flesch_ease', flesche_ease_udf(col("_Body")))
print(metrics.show())

# word_count_udf = udf(lambda x: word_count(x))
# post_df_small = post_df_small.withColumn('bodyWordCount', word_count_udf(col('_Body')))
# print(post_df_small.show())
