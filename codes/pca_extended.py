"""
SCRIPT for translating questions into metrics, similar to metrics_calculator.py. These are then saved(written to disk) and used in PCA Analysis
We only scan questions, and not their respective answers
"""
# time spark-submit   --conf "spark.pyspark.python=./env1/bin/python" --conf "spark.pyspark.driver.python=./env1/bin/python" pca_extended.py
# PYSPARK_PYTHON=./env1/bin/python spark-submit --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./env1/bin/python --conf spark.dynamicAllocation.maxExecutors=30 --conf spark.executor.memoryOverhead=4096 --master yarn --archives hdfs:///user/s1939882/env1.tar.gz#env1 codes/pca_extended.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, to_timestamp, year, datediff, hour, minute, unix_timestamp

from pyspark.sql.types import ArrayType, DoubleType, StringType
import re

import nltk
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer as sia
from nltk import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
# from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
# from pyspark.ml.stat import Correlation
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
nltk.data.path.append('/stackoverflow_env/stackoverflow_env/nltk_data')
nltk.data.path.append('/env1/env1/nltk_data')
nltk.data.path.append('/env1/nltk_data')

RULE_CODES = re.compile('<code>.*?</code>', re.DOTALL)
PCA_PATH = "pca_whole/"
years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
YEAR = 2015
FILE_SAVE_NAME = "metrics_questions_" + str(YEAR)



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


def calculate_metrics(original_body: str, original_title: str, original_tags: str) -> list:
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
    cleaned_tags = clean_tags(original_tags)
    syllables = syllable_count(cleaned_text)
    words = word_count(cleaned_text)
    characters = char_count(cleaned_text)
    sentences = sentence_count(cleaned_text)
    if words > 10:
        flesch_kincaid_grade = flesch_grade(syllables, sentences, words)
        flesch_reading_ease = flesch_ease(syllables, sentences, words)
        coleman_liau_index = coleman_liau(characters, sentences, words)
        code_percentages = code_percentage(original_body, characters)
        sentiment = sia().polarity_scores(cleaned_text)['compound']
        cosine_similarity_metrics_post_title = calculate_cosine_similarity(cleaned_title, cleaned_text)
        cosine_similarity_metrics_tags_title = calculate_cosine_similarity(cleaned_tags, cleaned_title)
        return [flesch_kincaid_grade, flesch_reading_ease, coleman_liau_index, code_percentages,
                cosine_similarity_metrics_post_title, cosine_similarity_metrics_tags_title, sentiment]
    return [0] * 7


#Big data of file size 37.9 GB
data_path = 'hdfs:///user/s2096307/posts.parquet'
ignore_cols_posts = ("_LastEditDate", "_OwnerUserId",
                     "_LastEditorUserId", "_LastEditorDisplayName",
                     "_LastEditDate", "_LastActivityDate", "_ContentLicense",
                     "_CommunityOwnedDate")
posts_df = spark.read.parquet(data_path).drop(*ignore_cols_posts)

# questions = posts_df.na.drop(subset=["_AcceptedAnswerId"])
questions = posts_df.filter(col('_PostTypeId') == 1)

# answers = posts_df.filter(col('_PostTypeId') == 2).select(col('_Id'), col('_Body'), col('_CreationDate')). \
#     withColumnRenamed('_Body', 'AcceptedAnswerText'). \
#     withColumnRenamed('_CreationDate', 'AcceptedAnswerCreationDate'). \
#     withColumnRenamed('_Id', 'AnswerId')

calculate_metrics_udf = \
    udf(lambda body, title, tags: calculate_metrics(body, title, tags), ArrayType(DoubleType()))
clean_tags_udf = udf(lambda tags: clean_tags_list(tags), ArrayType(StringType()))

# Filter the year, to subset the data
questions = questions.withColumn("_CreationDate", to_timestamp("_CreationDate")). \
    withColumn("Year", year("_CreationDate")).filter(col("Year") > YEAR)
#
# print("Questions in year: ", str(YEAR), "COUNT: ", questions.count())
questions_without_ans = questions.withColumn("TagsList", clean_tags_udf(col('_Tags'))). \
    withColumn('metrics',
               calculate_metrics_udf(col('_Body'), col('_Title'), col('_Tags'))). \
    withColumn('Flesch_Kincard_Grade', col('metrics')[0]). \
    withColumn('Flesch_reading_ease', col('metrics')[1]). \
    withColumn('Coleman_Liau_index', col('metrics')[2]). \
    withColumn('code_percentage', col('metrics')[3]). \
    withColumn('cos_sim_post_title', col('metrics')[4]). \
    withColumn('cos_sim_title_tag', col('metrics')[5]). \
    withColumn('sentiment', col('metrics')[6]). \
    drop('_Body', '_Id')

# no repartition
metrics = questions_without_ans
# 23.273.009 questions
# 3.367.908 unanswered questions
# 11.859.094 with accepted answer
# 11.413.915 answered questions

"""
Distribution of questions per year
|year|  count|
+----+-------+
|2018|1.888.484|
|2015|2.196.057|
|2022|1.585.489|
|2013|2.032.866|
|2014|2.135.338|
|2019|1.766.329|
|2020|1.870.637|
|2012|1.629.039|
|2009|  341.574|
|2016|2.200.069|
|2010|  690.700|
|2011|1.189.617|
|2008|   57.541|
|2017|2.115.639|
|2021|1.573.630|
+----+-------+"""

dataset = metrics.select("_Score",
                         "_ViewCount",
                         "_AnswerCount",
                         "_CommentCount",
                         "_FavoriteCount",
                         "_AcceptedAnswerId",
                         metrics.metrics[0].alias('FG'),
                         metrics.metrics[1].alias('FE'),
                         metrics.metrics[2].alias('CL'),
                         metrics.metrics[3].alias('CP'),
                         metrics.metrics[4].alias('CSPT'),
                         metrics.metrics[5].alias('CSTT'),
                         metrics.metrics[6].alias('SENT'))

dataset = dataset.withColumn('FG', col("FG").cast('double'))
dataset = dataset.withColumn('FE', col("FE").cast('double'))
dataset = dataset.withColumn('CL', col("CL").cast('double'))
dataset = dataset.withColumn('CP', col("CP").cast('double'))
dataset = dataset.withColumn('CSPT', col("CSPT").cast('double'))
dataset = dataset.withColumn('CSTT', col("CSTT").cast('double'))
dataset = dataset.withColumn('SENT', col("SENT").cast('double'))
dataset = dataset.withColumn('_AnswerCount', col('_AnswerCount').cast('double'))
dataset = dataset.withColumn('_ViewCount', col('_ViewCount').cast('double'))
dataset = dataset.withColumn('_CommentCount', col('_CommentCount').cast('double'))
dataset = dataset.withColumn('_Score', col('_Score').cast('double'))
dataset = dataset.withColumn('_FavoriteCount', col('_FavoriteCount').cast('double'))
dataset = dataset.fillna(0)

dataset = dataset.withColumn('target',
                             F.when((col("_AnswerCount") > 0) & (col("_AcceptedAnswerId").isNull()), F.lit("Answered")). \
                             when((col("_AnswerCount") > 0) & (col("_AcceptedAnswerId") > 0), F.lit("BestAnswered")). \
                             when(col("_AnswerCount") == 0, F.lit("Unanswered")))

# Since all data would only contain 100,000 rows of only digits, this is relatively small data and thus can be saved
dataset.repartition(5).write.mode('overwrite').parquet(FILE_SAVE_NAME)

# Next, in order to train ML models in Spark later, we'll use the VectorAssembler to combine a given list of columns into a single vector column.
# assembler = VectorAssembler(
#     inputCols=['FG', 'FE', 'CL', 'CP', 'CSPT', 'CSTT', "SENT"], outputCol='features') #, "_AnswerCount", "_ViewCount", "_FavoriteCount", "_CommentCount"
# df = assembler.transform(dataset).select('features')
#
# #get correlation matrix
# matrix = Correlation.corr(df, "features")
# #If you want to get the result as a numpy array (on your driver), you can use the following:
# print(matrix.collect()[0]["pearson({})".format("features")].values)

#
# # print(df.show(6))
# print("VECTOR CREATED")
# # Next, we standardize the features, notice here we only need to specify the assembled column as the input feature
# scaler = StandardScaler(
#     inputCol='features',
#     outputCol='scaledFeatures',
#     withMean=True,
#     withStd=True
# ).fit(df)
#
# df_scaled = scaler.transform(df)
# print("SCALING DONE")
#
# # After the preprocessing step, we fit the PCA model.
# n_components = 2
# pca = PCA(
#     k=n_components,
#     inputCol='scaledFeatures',
#     outputCol='pcaFeatures'
# ).fit(df_scaled)
#
# df_pca = pca.transform(df_scaled)
# print('Explained variance Ratio', pca.explainedVariance.toArray())
#
#
# pca.save(PCA_PATH)
# print("saved succesfully")
