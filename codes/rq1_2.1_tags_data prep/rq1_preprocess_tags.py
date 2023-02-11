'''
Research question 1: Analysing tags and posts
This script is use to pre-process tags from posts table.
Run this script in the cluster using following command:
time spark-submit --master yarn \
    --deploy-mode cluster \
    --conf spark.dynamicAllocation.maxExecutors=15 \
    --conf spark.dynamicAllocation.minExecutors=10 \
    --conf spark.dynamicAllocation.initialExecutors=5 \
    --jars /home/s2812940/spark-xml_2.11-0.9.0.jar preprocess_tags.py 2> /dev/null

Pyspark package:
pyspark --jars /home/s2812940/spark-xml_2.11-0.9.0.jar

Output to parquets:
- tags_questions: tags after cleaning. Only SO questions (PostTypeId=1) has tags. 
- tags_distribution: tags after cleaning, exploded at word level. This dataset can be used for ML tasks such as clustering (Column 'Tag' is a corpus),
or visualization purposes such as distribution of some metrics such as viewcount, answer count for top tags, etc.
- top_tags: tags are aggregated based on tag frequency (count), median viewcount, median answercount
- tags_num: number of tags per question and how it relate to possibility of getting answer => This is for RQ 2.1.1
'''

from pyspark import SparkContext
from pyspark.sql import SQLContext

import sys
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as F

#sc.stop()
sc = SparkContext(appName="stack_exchange")
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()

# Read data
# tags = spark.read.parquet('/user/s2812940/tags_sample.parquet')
posts = spark.read.parquet('/user/s2812940/project/parquet_data/posts.parquet')
posts.count()

# PRE-PROCESS DATA --------------------------

# Filter questions from posts
col_list = ['_Id', '_PostTypeId', '_AcceptedAnswerId', '_CreationDate', '_ViewCount', '_Score', '_Tags', '_AnswerCount']
df_questions = (
    posts.select(col_list)
    .filter(col('_PostTypeId') == 1) # filter questions
    .withColumn("_CreationDate", to_timestamp("_CreationDate")) # Convert _CreationDate to timestampt
    .withColumn('is_answer', F.when(F.col('_AcceptedAnswerId').isNull(), 0).otherwise(1)) # Create is_answer column
)
num_questions = df_questions.count()

# Create Year and Month based on '_CreationDate'
df_questions = (df_questions
                    .withColumn("Year", year("_CreationDate"))
                    .withColumn("Month", month("_CreationDate"))
                    .drop('_CreationDate') # drop _CreationDate
                )

# Clean tags, number of tags per post
df_questions = (df_questions
.withColumn('_Tags', split('_Tags', '><'))
.selectExpr('*', "TRANSFORM(_Tags, value -> regexp_replace(value, '(>|<)', '')) AS tags_arr") # Clean tags, create array of tags
.withColumn('TagCountPerPost', size(array_distinct('tags_arr'))) # count number of tags per post
)

# df_questions.show()
df_questions.repartition(10).write.parquet('tags_questions.parquet') # Write parquet data to reuse later

# Explode by tag
df_questions = df_questions.withColumn('Tag', explode('tags_arr')).drop('_Tags').drop('tags_arr')

# AGGREGATION FOR FURTHER ANALYSIS --------------------------
# Tags distribution table
drop_col = ('_Tags', 'tags_arr', '_PostTypeId', '_AcceptedAnswerId')

#tags_distribution = df_questions.withColumn('Tag', explode('tags_arr')).drop('_Tags').drop('tags_arr').drop(*drop_col)
tags_distribution = df_questions.drop(*drop_col)
# tags_distribution.show()
tags_distribution.count()
tags_distribution.repartition(10).write.parquet('tags_distribution.parquet') # Write parquet

# Aggregation: top_tags table
top_tags = (
tags_distribution.groupBy('Tag') 
.agg(
    count('Tag').alias('tag_frequency'), # count tags
    F.expr('percentile(_Score, 0.5)').alias('med_score'), # median of score
    F.expr('percentile(_AnswerCount, 0.5)').alias('med_answer_count'), # median of answer count
    (sum('is_answer')/count('Tag')).alias('percent_answer') # percentage of getting answer
).orderBy(desc('tag_frequency'))
)
# top_tags.show()
top_tags.count()
top_tags.repartition(10).write.parquet('top_tags.parquet') # write parquet

# Aggregation: tags_num table:
# Summary by TagCountPerPost (1,2,..6), show percentage get answered, percentage of all questions

tags_num = (df_questions
.groupBy('TagCountPerPost')
.agg(
    round((count('_PostTypeId')/num_questions), 4).alias('percent_all'),
    round(sum('is_answer')/count('_PostTypeId'), 4).alias('percent_answer')
))

tags_num.show()
tags_num.write.parquet('tags_num.parquet')
