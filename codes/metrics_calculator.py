'''
LOCAL RUNNING:
time spark-submit   --conf "spark.pyspark.python=./../../miniconda3/envs/bigdataEnv/bin/python" --conf "spark.pyspark.driver.python=../../miniconda3/envs/bigdataEnv/bin/python" metrics_calculator.py
CLUSTER RUNNING:
# PYSPARK_PYTHON=./MBD-env/MBD-env/bin/python spark-submit --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./MBD-env/MBD-env/bin/python --master yarn --deploy-mode cluster --archives MBD-env.zip#MBD-env metrics_calculator.py'''
# from __future__ import division # THIS LINE IS FUCKING IMPORTANT FOR PYTHON 2 ONLY!!!!!


from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_timestamp, year, datediff, hour, minute, unix_timestamp
import re
from pyspark.sql.types import ArrayType, DoubleType, StringType
# import nltk
# from nltk.sentiment.util import *
# from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# sc = SparkContext(appName="stack_exchange")
# sc.setLogLevel("ERROR")
# sqlContext = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()


# spark.sparkContext.addFile('hdfs:///user/s2096307/nltk_data', recursive=True)
# nltk_path = SparkFiles.get('nltk_data')
# print(nltk_path)
# nltk.data.path.append('nltk_data/nltk_data')
# nltk.download('vader_lexicon', download_dir=nltk_path)

def clean_data(text, codeless=True):
    rules = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    text = text.lower()
    if codeless:
        text = re.sub("<code>.*?</code>", "", text)  # remove code block
    cleantext = re.sub(rules, '', text)
    return cleantext


def word_count(content):
    return len(content.split())


def char_count(content):
    return len(content)


def sentence_count(content):
    sentences = 0
    seen_end = False
    sentence_end = {'?', '!', '.', '...'}
    for c in content:
        if c in sentence_end:
            if not seen_end:
                seen_end = True
                sentences += 1
            continue
        seen_end = False
    return sentences if sentences != 0 else 1


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
    if no_words == 0:
        return None
    L = (int(no_chars) / int(no_words)) * 100
    S = (int(no_sentences) / int(no_words)) * 100
    return 0.0588 * L - 0.296 * S - 15.8


def flesch_ease(no_syllables, no_sentences, no_words):
    if no_sentences == 0 or no_words == 0:
        return None
    words_per_sent = no_words / no_sentences
    syll_per_word = no_syllables / no_words
    return 206.835 - (1.015 * words_per_sent) - (84.6 * syll_per_word)


def flesch_grade(no_syllables, no_sentences, no_words):
    if no_sentences == 0 or no_words == 0:
        return None
    words_per_sent = no_words / no_sentences
    syll_per_word = no_syllables / no_words
    return 0.39 * words_per_sent + 11.8 * syll_per_word - 15.59


def code_percentage(original_body, no_char):
    code_text_list = re.findall("<code>.*?</code>", original_body)  # remove code block
    if code_text_list:
        code_texts = ''
        for code_text in code_text_list:
            code_texts += code_text
        code_texts = re.sub("<code>", '', code_texts)
        code_texts = re.sub("</code>", '', code_texts)
        code_chars = len(code_texts)
        return code_chars / no_char
    else:
        return 0


def calculate_cosine_similarity(body_1, body_2):
    corpus = [body_1, body_2]
    trsfm = TfidfVectorizer().fit_transform(corpus)
    return cosine_similarity(trsfm[0], trsfm).tolist()[0][1]


def clean_tags(raw_tags):
    tag_lists = raw_tags.split('><')
    tag_str = ''
    for i in range(len(tag_lists)):
        tag_lists[i] = re.sub('>', '', tag_lists[i])
        tag_lists[i] = re.sub('<', '', tag_lists[i])
        tag_str += tag_lists[i] + ' '
    return tag_str


def clean_tags_list(raw_tags):
    tag_lists = raw_tags.split('><')
    tag_lists_cleaned = []
    for i in range(len(tag_lists)):
        tag_lists[i] = re.sub('>', '', tag_lists[i])
        tag_lists[i] = re.sub('<', '', tag_lists[i])
        tag_lists_cleaned.append(tag_lists[i])
    return tag_lists_cleaned


def calculate_metrics(original_body, original_title, original_tags, original_answer):
    cleaned_text = clean_data(original_body)
    cleaned_title = clean_data(original_title)
    cleaned_tags = clean_tags(original_tags)
    syllables = syllable_count(cleaned_text)
    words = word_count(cleaned_text)
    characters = char_count(cleaned_text)
    sentences = sentence_count(cleaned_text)
    flesch_kincaid_grade = flesch_grade(syllables, sentences, words)
    flesch_reading_ease = flesch_ease(syllables, sentences, words)
    coleman_liau_index = coleman_liau(characters, sentences, words)
    code_percentages = code_percentage(original_body, char_count(clean_data(original_body, codeless=False)))
    # sentiment = sia().polarity_scores(cleaned_text)['compound']
    cosine_similarity_metrics_post_title = calculate_cosine_similarity(cleaned_title, cleaned_text)
    cosine_similarity_metrics_tags_title = calculate_cosine_similarity(cleaned_tags, cleaned_title)
    return [flesch_kincaid_grade, flesch_reading_ease, coleman_liau_index, code_percentages,
            cosine_similarity_metrics_post_title, cosine_similarity_metrics_tags_title]


data_path = "posts.parquet/part-00006-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet"
posts_df = spark.read.parquet(data_path)
ignore_cols_posts = ("_LastEditDate", "_OwnerUserId",
                     "_LastEditorUserId", "_LastEditorDisplayName",
                     "_LastEditDate", "_LastActivityDate", "_ContentLicense",
                     "_CommunityOwnedDate")

questions = posts_df.na.drop(subset=["_AcceptedAnswerId"])

answers = posts_df.filter(col('_PostTypeId') == 2).select(col('_Id'), col('_Body'), col('_CreationDate')). \
    withColumnRenamed('_Body', 'AcceptedAnswerText'). \
    withColumnRenamed('_CreationDate', 'AcceptedAnswerCreationDate'). \
    withColumnRenamed('_Id', 'AnswerId')

calculate_metrics_udf = \
    udf(lambda body, title, tags, answer: calculate_metrics(body, title, tags, answer), ArrayType(StringType()))
clean_tags_udf = udf(lambda tags: clean_tags_list(tags), ArrayType(StringType()))

# years = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
# for year in years:
#     questions_year = questions.
#     questions = questions.merge()

questions = questions.join(answers, questions._AcceptedAnswerId == answers.AnswerId). \
    drop(*ignore_cols_posts). \
    withColumn("_CreationDate", to_timestamp("_CreationDate")). \
    withColumn("Year", year("_CreationDate")). \
    filter(col('Year') > 2020). \
    drop(col('Year')). \
    withColumn("AcceptedAnswerCreationDate", to_timestamp("AcceptedAnswerCreationDate")). \
    withColumn("TagsList", clean_tags_udf(col('_Tags'))).\
    withColumn('metrics',
               calculate_metrics_udf((col('_Body')), col('_Title'), col('_Tags'), col('AcceptedAnswerText'))). \
    withColumn('AnswerDurationSeconds',
               unix_timestamp(col('AcceptedAnswerCreationDate')) - unix_timestamp(col('_CreationDate'))). \
    withColumn('AnswerDurationDays', datediff(col('AcceptedAnswerCreationDate'), col('_CreationDate'))). \
    withColumn('AnswerDurationMinute', col('AnswerDurationSeconds') / 60). \
    withColumn('AnswerDurationHour', col('AnswerDurationMinute') / 60). \
    withColumn('Flesch_Kincard_Grade', col('metrics')[0]). \
    withColumn('Flesch_reading_ease', col('metrics')[1]). \
    withColumn('Coleman_Liau_index', col('metrics')[2]). \
    withColumn('code_percentage', col('metrics')[3]). \
    withColumn('cos_sim_post_title', col('metrics')[4]). \
    withColumn('cos_sim_title_tag', col('metrics')[5])

questions.repartition(10).write.mode('overwrite').parquet('questions-with-ans-and-metrics-cluster-small.parquet')

# spark-submit   --conf "spark.pyspark.python=./MBD-env/bin/python" metrics_calculator.py
