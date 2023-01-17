"""
Last modified on 16th Jan.
Author: Hung.

time spark-submit --master yarn --deploy-mode cluster  create_parquet.py

"""

from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
import re
from pyspark.sql.functions import udf, col


def word_count(content):
    rules = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    cleantext = re.sub(rules, '', content.lower())
    cleantext = re.sub('[^a-z0-9]+', ' ', cleantext)
    text_length = len(cleantext.split())

    return text_length

# SUbjectivity
# Sentiment
# Colemann

sc = SparkContext(appName="stack_exchange")
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()


post_df = spark.read.parquet('/user/s2812940/project/parquet_data/posts.parquet/part-00000-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet')
# Read a fraction of data (100 rows). Apply lambda function to count words.
post_df_small = post_df.limit(1)

word_count_udf = udf(lambda x: word_count(x))
post_df_small = post_df_small.withColumn('bodyWordCount', word_count_udf(col('_Body')))
print(post_df_small.show())







