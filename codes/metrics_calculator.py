from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
import re
import nltk
from nltk.tokenize import sent_tokenize

'''
LOCAL RUNNING: 
time spark-submit   --conf "spark.pyspark.python=./../../miniconda3/envs/bigdataEnv/bin/python" --conf "spark.pyspark.driver.python=../../miniconda3/envs/bigdataEnv/bin/python" metrics_calculator.py 
CLUSTER RUNNING: 
PYSPARK_PYTHON=./envs/MBD-stackoverflow/bin/python spark-submit --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./envs/MBD-stackoverflow/bin/python --master yarn-cluster --archives envs/MBD-stackoverflow.zip#MBD-stackoverflow metrics_calculator.py

'''

spark = SparkSession.builder.appName("Colyn").getOrCreate()


def clean_data(text):
    rules = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = text.lower()
    codeless_text = re.sub("<code>.*?</code>", "", text)  # remove code block
    cleantext = re.sub(rules, '', codeless_text)
    return cleantext


def word_count(content):
    return len(content.split())


def char_count(content):
    return len(content)


def sentence_count(content):
    return len(sent_tokenize(content))


def syllable_count(text):
    count = 0
    for word in text.split():
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


def coleman_liau(no_chars, no_sentences, no_words):
    L = (int(no_chars) / int(no_words)) * 100
    S = (int(no_sentences) / int(no_words)) * 100
    return 0.0588 * L - 0.296 * S - 15.8

def flesch_reading_ease(no_words, no_sentences, no_syllables):
    return


small_data_path = "posts.parquet/part-00000-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet"
posts_df = spark.read.parquet(small_data_path)
ignore_cols_posts = ("_LastEditDate", "_OwnerUserId",
                     "_LastEditorUserId", "_LastEditorDisplayName",
                     "_LastEditDate", "_LastActivityDate", "_ContentLicense",
                     "_CommunityOwnedDate")

questions = posts_df.na.drop(subset=["_AcceptedAnswerId"])
questions = questions.drop(*ignore_cols_posts)

questions = questions.filter(col('_Id') < 7414409)

word_count_udf = udf(lambda text: word_count(text))
clean_data_udf = udf(lambda raw_data: clean_data(raw_data))
sentence_count_udf = udf(lambda text: sentence_count(text))
syllable_count_udf = udf(lambda text: syllable_count(text))
char_count_udf = udf(lambda text: char_count(text))
cli_udf = udf(lambda chars, sentences, words: coleman_liau(chars, sentences, words))

questions = questions.withColumn('cleaned_body', clean_data_udf(col('_Body'))) \
    .withColumn('wordCount', word_count_udf(col('cleaned_body')))\
    .withColumn('sentenceCount', sentence_count_udf(col('cleaned_body')))\
    .withColumn('syllableCount', syllable_count_udf(col('cleaned_body')))\
    .withColumn('charCount', char_count_udf(col('cleaned_body')))\
    .withColumn('CLI', cli_udf(col('charCount'), col('sentenceCount'), col('wordCount')))

# questions.write.parquet("questions_small.parquet")
questions.show()
