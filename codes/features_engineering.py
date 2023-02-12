"""
LOCAL RUNNING (NOT RECOMMENDED):
time spark-submit   --conf "spark.pyspark.python=./../../miniconda3/envs/bigdataEnv/bin/python" \\
--conf "spark.pyspark.driver.python=../../miniconda3/envs/bigdataEnv/bin/python" features_engineering.py

CLUSTER RUNNING:

PYSPARK_PYTHON=./stackoverflow_env/stackoverflow_env/bin/python spark-submit --conf \\
spark.dynamicAllocation.maxExecutors=20 --conf \\
spark.yarn.appMasterEnv.PYSPARK_PYTHON=./stackoverflow_env/stackoverflow_env/bin/python --master yarn --deploy-mode \\
cluster --archives stackoverflow_env.zip#stackoverflow_env  features_engineering.py
With stackoverflow_env.zip is the zip folder contains the conda environment with needed libraries,
including scikit-learn, nltk, and their dependencies.

"""
# from __future__ import division # THIS LINE IS FUCKING IMPORTANT FOR PYTHON 2 ONLY!!!!!
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_timestamp, year, datediff, hour, minute, unix_timestamp, pandas_udf
import re
from pyspark.sql.types import ArrayType, DoubleType, StringType
# import nltk
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
import nltk
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
spark = SparkSession.builder.getOrCreate()
nltk.data.path.append('/stackoverflow_env/stackoverflow_env/nltk_data')  # Path for NLTK Data used: Vader package

RULE_CODES = re.compile('<code>.*?</code>', re.DOTALL)  # Regular expressions for extracting paragraph of code.


def clean_data(text: str, codeless=True) -> str:
    """
    Remove HTML tags, code tags (if desired), etc. Make text lowercase.
    :param text: Raw text paragraph.
    :param codeless: Set to true will return cleaned text without HTML <code> blocks.
    :return: Cleaned text.
    """
    rules = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    # text = text.lower()
    if codeless:
        text = re.sub(RULE_CODES, "", text)  # remove code block
    cleantext = re.sub(rules, '', text)
    return cleantext


def word_count(content: str) -> int:
    """
    Count the number of words in the content
    :param content: Text content
    :return: number of words in content.
    """
    return len(content.split())


def char_count(content: str) -> int:
    """
    Count the number of characters in the content
    :param content: Text content
    :return: number of characters in content.
    """
    return len(content)


def sentence_count(content: str) -> int:
    """
    Count the number of sentences in the content using NLTK's sent_tokenize.
    :param content: Text content
    :return: number of sentences in content.
    """
    return len(sent_tokenize(content))


def syllable_count(text: str) -> int:
    """
    Count the number of syllable in the content.
    :param text: Text content:
    :return: Number of syllables.
    """
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


def coleman_liau(no_chars: int, no_sentences: int, no_words: int) -> float:
    """
    Calculate the Coleman-Liau index for a paragraph.
    :param no_chars: Number of characters.
    :param no_sentences: Number of sentences.
    :param no_words: Number of words.
    :return: Coleman-Liau index.
    """
    if no_words == 0:
        return -999
    L = (int(no_chars) / int(no_words)) * 100
    S = (int(no_sentences) / int(no_words)) * 100
    return 0.0588 * L - 0.296 * S - 15.8


def flesch_ease(no_syllables: int, no_sentences: int, no_words: int) -> float:
    """
    Calculate the Flesch Reading Ease score for a paragraph.
    :param no_syllables: Number of characters.
    :param no_sentences: Number of sentences.
    :param no_words: Number of words.
    :return: Flesch Reading Ease score
    """
    if no_sentences == 0 or no_words == 0:
        return -999
    words_per_sent = no_words / no_sentences
    syll_per_word = no_syllables / no_words
    return 206.835 - (1.015 * words_per_sent) - (84.6 * syll_per_word)


def flesch_grade(no_syllables: int, no_sentences: int, no_words: int) -> float:
    """
    Calculate the Flesch-Kincaid score for a paragraph.
    :param no_syllables: Number of characters.
    :param no_sentences: Number of sentences.
    :param no_words: Number of words.
    :return: Flesch-Kincaid score
    """
    if no_sentences == 0 or no_words == 0:
        return -999
    words_per_sent = no_words / no_sentences
    syll_per_word = no_syllables / no_words
    return 0.39 * words_per_sent + 11.8 * syll_per_word - 15.59


def code_percentage(original_body: str, no_char: int) -> float:
    """
    Calculate how many percent of post's character is inside the code block.
    :param original_body: Original text (text that have not been cleaned).
    :param no_char: Number of characters in the post.
    :return: Percentage of code character.
    """
    code_text_list = re.findall(RULE_CODES, original_body)
    if code_text_list:
        code_texts = ''
        for code_text in code_text_list:
            code_texts += code_text
        code_texts = re.sub("<code>", '', code_texts)
        code_texts = re.sub("</code>", '', code_texts)
        code_chars = len(code_texts)
        return code_chars / (no_char + code_chars)
    else:
        return 0


def calculate_cosine_similarity(body_1: str, body_2: str) -> float:
    """
    Calculate the cosine similarity for two text bodies using scikit-learn's cosine_similarity and TfidfVectorizer.
    :param body_1: Text body 1.
    :param body_2: Text body 2.
    :return: Cosine Similarity Scores.
    """
    if not body_1 or not body_2:
        return 0
    try:
        corpus = [body_1, body_2]
        transform_tfidf = TfidfVectorizer().fit_transform(corpus)
        return cosine_similarity(transform_tfidf[0], transform_tfidf).tolist()[0][1]
    except ValueError:
        return 0


def clean_tags(raw_tags: str) -> str:
    """
    Remove parentheses of tags list and append to a string.
    :param raw_tags: Original tag lists with <> parentheses.
    :return: A list of tags formatted as string, seperated by space.
    """
    tag_lists = raw_tags.split('><')
    tag_str = ''
    for i in range(len(tag_lists)):
        tag_lists[i] = re.sub('>', '', tag_lists[i])
        tag_lists[i] = re.sub('<', '', tag_lists[i])
        tag_str += tag_lists[i] + ' '
    return tag_str


def clean_tags_list(raw_tags: str) -> list:
    """
    Remove parentheses of tags list and append to a list.
    :param raw_tags: Original tag lists with <> parentheses.
    :return: A list of tags.
    """

    tag_lists = raw_tags.split('><')
    tag_lists_cleaned = []
    for i in range(len(tag_lists)):
        tag_lists[i] = re.sub('>', '', tag_lists[i])
        tag_lists[i] = re.sub('<', '', tag_lists[i])
        tag_lists_cleaned.append(tag_lists[i])
    return tag_lists_cleaned


def calculate_metrics(original_body, original_title, original_tags):
    """
    Calculate the defined metrics: Flesc    h-Kincaid grade, Flesch Reading Ease, Coleman-Liau index, sentiment of post,
    cosine similarity between post and title/tag and title, and code percentage.
    :param original_body: Original (uncleaned body of post).
    :param original_title: Original (uncleaned title of post).
    :param original_tags: Original (uncleaned tags of post).
    :return: A list of all metric scores.
    """
    cleaned_text = clean_data(original_body)
    cleaned_title = clean_data(original_title)
    cleaned_tags = clean_data(original_body)
    syllables = syllable_count(original_tags)
    words = word_count(cleaned_text)
    characters = char_count(cleaned_text)
    sentences = sentence_count(cleaned_text)
    if words > 10:
        flesch_kincaid_grade = flesch_grade(syllables, sentences, words)
        flesch_reading_ease = flesch_ease(syllables, sentences, words)
        coleman_liau_index = coleman_liau(characters, sentences, words)
        code_percentages = code_percentage(original_body.str, characters)
        sentiment = sia().polarity_scores(cleaned_text)['compound']
        cosine_similarity_metrics_post_title = calculate_cosine_similarity(cleaned_title, cleaned_text)
        cosine_similarity_metrics_tags_title = calculate_cosine_similarity(cleaned_tags, cleaned_title)
        return [flesch_kincaid_grade, flesch_reading_ease, coleman_liau_index, code_percentages,
                cosine_similarity_metrics_post_title, cosine_similarity_metrics_tags_title, sentiment]
    return [0] * 7


def calculate_metrics_pandas(original_body, original_title, original_tags):
    """
    Calculate the defined metrics for pandas udf: Flesch-Kincaid grade, Flesch Reading Ease, Coleman-Liau index, sentiment of post,
    cosine similarity between post and title/tag and title, and code percentage.
    :param original_body: Original (uncleaned body of post).
    :param original_title: Original (uncleaned title of post).
    :param original_tags: Original (uncleaned tags of post).
    :return: A list of all metric scores.
    """
    cleaned_text = original_body.apply(lambda s: clean_data(s))
    cleaned_title = original_title.apply(lambda s: clean_data(s))
    cleaned_tags = original_tags.apply(lambda s: clean_data(s))
    syllables = cleaned_text.apply(lambda s: syllable_count(s))
    words = cleaned_text.apply(lambda s: word_count(s))
    characters = cleaned_text.apply(lambda s: char_count(s))
    sentences = cleaned_text.apply(lambda s: sentence_count(s))
    if words > 10:
        flesch_kincaid_grade = flesch_grade(syllables, sentences, words)
        flesch_reading_ease = flesch_ease(syllables, sentences, words)
        coleman_liau_index = coleman_liau(characters, sentences, words)
        code_percentages = code_percentage(original_body.str, characters)
        sentiment = sia().polarity_scores(cleaned_text)['compound']
        cosine_similarity_metrics_post_title = calculate_cosine_similarity(cleaned_title, cleaned_text)
        cosine_similarity_metrics_tags_title = calculate_cosine_similarity(cleaned_tags, cleaned_title)
        return [flesch_kincaid_grade, flesch_reading_ease, coleman_liau_index, code_percentages,
                cosine_similarity_metrics_post_title, cosine_similarity_metrics_tags_title, sentiment]
    return [0] * 7



data_path = ["posts.parquet/part-00000-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet",]
             # "posts.parquet/part-00001-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet",
             # "posts.parquet/part-00002-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet",
             # "posts.parquet/part-00003-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet",
             # "posts.parquet/part-00004-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet"]

suuper_big_data_path = 'posts.parquet'
ignore_cols_posts = ("_LastEditDate", "_OwnerUserId",
                     "_LastEditorUserId", "_LastEditorDisplayName",
                     "_LastEditDate", "_LastActivityDate", "_ContentLicense",
                     "_CommunityOwnedDate")
posts_df = spark.read.parquet(*data_path).drop(*ignore_cols_posts)

# questions = posts_df.na.drop(subset=["_AcceptedAnswerId"])
questions = posts_df.filter(col('_PostTypeId') == 1)

answers = posts_df.filter(col('_PostTypeId') == 2).select(col('_Id'), col('_CreationDate')). \
    withColumnRenamed('_CreationDate', 'AcceptedAnswerCreationDate'). \
    withColumnRenamed('_Id', 'AnswerId')

calculate_metrics_udf = \
    udf(lambda body, title, tags: calculate_metrics(body, title, tags), ArrayType(StringType()))

calculate_metrics_pandas_udf = \
    pandas_udf(lambda body, title, tags: calculate_metrics_pandas(body, title, tags), ArrayType(StringType()))

questions = questions.withColumn("_CreationDate", to_timestamp("_CreationDate")). \
    withColumn("Year", year("_CreationDate"))

years = [2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
# new_questions = questions.filter(col("Year") == 2008) \
#     .sample(fraction=0.1)
# for year in years:
#     questions_year = questions.filter(col("Year") == year) \
#         .sample(fraction=0.1)
#     new_questions = new_questions.union(questions_year)
# questions = new_questions
answers = answers.repartition(256, col('AnswerId'))

# questions = questions.join(answers, questions._AcceptedAnswerId == answers.AnswerId, 'leftouter'). \
#     withColumn("AcceptedAnswerCreationDate", to_timestamp("AcceptedAnswerCreationDate")). \
#     withColumn('metrics',
#                calculate_metrics_udf((col('_Body')), col('_Title'), col('_Tags'))). \
#     withColumn('AnswerDurationSeconds',
#                unix_timestamp(col('AcceptedAnswerCreationDate')) - unix_timestamp(col('_CreationDate'))). \
#     withColumn('AnswerDurationDays', datediff(col('AcceptedAnswerCreationDate'), col('_CreationDate'))). \
#     withColumn('AnswerDurationMinute', col('AnswerDurationSeconds') / 60). \
#     withColumn('AnswerDurationHour', col('AnswerDurationMinute') / 60). \
#     withColumn('Flesch_Kincard_Grade', col('metrics')[0]). \
#     withColumn('Flesch_reading_ease', col('metrics')[1]). \
#     withColumn('Coleman_Liau_index', col('metrics')[2]). \
#     withColumn('code_percentage', col('metrics')[3]). \
#     withColumn('cos_sim_post_title', col('metrics')[4]). \
#     withColumn('cos_sim_title_tag', col('metrics')[5]). \
#     withColumn('sentiment', col('metrics')[6]). \
#     drop('_Body', 'Year')

questions = questions.withColumn('metrics',
                                 calculate_metrics_pandas_udf((col('_Body')), col('_Title'), col('_Tags'))). \
    drop('_Body', 'Year'). \
    withColumn('Flesch_Kincard_Grade', col('metrics')[0]). \
    withColumn('Flesch_reading_ease', col('metrics')[1]). \
    withColumn('Coleman_Liau_index', col('metrics')[2]). \
    withColumn('code_percentage', col('metrics')[3]). \
    withColumn('cos_sim_post_title', col('metrics')[4]). \
    withColumn('cos_sim_title_tag', col('metrics')[5]). \
    withColumn('sentiment', col('metrics')[6]). \
    repartition(256, col('_AcceptedAnswerId'))

# questions = questions.repartition(60, col('_AcceptedAnswerId'))

questions = questions.join(answers, questions._AcceptedAnswerId == answers.AnswerId, 'leftouter'). \
    withColumn('AnswerDurationSeconds',
               unix_timestamp(col('AcceptedAnswerCreationDate')) - unix_timestamp(col('_CreationDate'))). \
    withColumn('AnswerDurationDays', datediff(col('AcceptedAnswerCreationDate'), col('_CreationDate'))). \
    withColumn('AnswerDurationMinute', col('AnswerDurationSeconds') / 60). \
    withColumn('AnswerDurationHour', col('AnswerDurationMinute') / 60)
# questions.write.mode('overwrite') \
# .parquet('postwithmetrics-smaller-08022023-20.parquet')
print('now to writing')
questions.write.mode('overwrite') \
    .parquet('test-all-halfdata.parquet')

# pyspark --conf "spark.pyspark.python=/usr/bin/python3.6"
