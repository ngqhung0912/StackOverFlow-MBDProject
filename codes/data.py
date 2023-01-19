# -*- coding: utf-8 -*-
# time spark-submit   --conf "spark.pyspark.python=../new_env/bin/python" --conf "spark.pyspark.driver.python=../new_env/bin/python" data.py
from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
import math
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from features import Features
from pyspark.sql.functions import udf, col

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

def flesche_grade(content):
    features = Features(content)
    return features.flesch_grade()

def coleman_liau(content):
    features = Features(content)
    return features.coleman_liau()

def code_percentage(content):
    features = Features(content)
    return features.code_percentage()


flesche_ease_udf = udf(lambda x: flesche_ease(x))
flesche_grade_udf = udf(lambda x: flesche_grade(x))
coleman_liau_udf = udf(lambda x: coleman_liau(x))
code_percentage_udf = udf(lambda x: code_percentage(x))
# Initialize a copy to do operations on and save later
metrics = questions.select(col("_AcceptedAnswerId"), col("_Body"))

metrics = metrics.withColumn('flesch_ease', flesche_ease_udf(col("_Body")))
metrics = metrics.withColumn('flesch_grade', flesche_grade_udf(col("_Body")))
metrics = metrics.withColumn('coleman', coleman_liau_udf(col("_Body")))
metrics = metrics.withColumn('code_percentage', code_percentage_udf(col("_Body")))
print(metrics.show())

# TODO: Write this to metrics.parquet and then use pca.py
# TODO: fix code_percentage code in features.py

