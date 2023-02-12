'''
Research question 2.1.3: Analysing tags and waiting time to get answered.

This script is use to prepare data for visualization: Aggregate tags based on their frequency,  
waiting time (in hours, days, seconds) to get answered for questions containing the tags, median of questions score, etc.

Input: Posts table (parquet files)
Output: top_tags_response.parquet

Run this script in the cluster using following command:
time spark-submit --master yarn \
    --deploy-mode cluster \
    --conf spark.dynamicAllocation.maxExecutors=15 \
    --conf spark.dynamicAllocation.minExecutors=10 \
    --conf spark.dynamicAllocation.initialExecutors=5 \
    --jars /home/s2812940/spark-xml_2.11-0.9.0.jar top_tags_response.py 2> /dev/null
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
posts = spark.read.parquet('/user/s2812940/project/parquet_data/posts.parquet')
posts.count()

# PRE-PROCESS DATA --------------------------
# FILTER ANSWER:
answers = posts.filter(col('_PostTypeId') == 2).select(col('_Id'), col('_Body'), col('_CreationDate')). \
    withColumnRenamed('_CreationDate', 'AcceptedAnswerCreationDate'). \
    withColumnRenamed('_Id', 'AnswerId')

answers.count() # 34,337,401 

# FILTER QUESTIONS
col_list = ['_Id', '_PostTypeId', '_AcceptedAnswerId', '_CreationDate', '_ViewCount', '_Score', '_Tags', '_AnswerCount']
questions = (
    posts.select(col_list)
    .withColumnRenamed('_Id', 'QuestionId')
    .filter(col('_PostTypeId') == 1) # filter questions
    .filter(F.col('_AcceptedAnswerId').isNull() == False) # filter questions get answer
)
num_questions = questions.count() # Total: 23,273,009  With answer: 11,859,094

# Merge question with answer:
df_question_with_answer = questions.join(answers, questions._AcceptedAnswerId == answers.AnswerId). \
    withColumn("_CreationDate", to_timestamp("_CreationDate")). \
    withColumn("Year", year("_CreationDate")). \
    withColumn("AcceptedAnswerCreationDate", to_timestamp("AcceptedAnswerCreationDate")). \
    withColumn('AnswerDurationSeconds',
               unix_timestamp(col('AcceptedAnswerCreationDate')) - unix_timestamp(col('_CreationDate'))). \
    withColumn('AnswerDurationDays', datediff(col('AcceptedAnswerCreationDate'), col('_CreationDate'))). \
    withColumn('AnswerDurationMinute', col('AnswerDurationSeconds') / 60). \
    withColumn('AnswerDurationHour', col('AnswerDurationMinute') / 60)

df_question_with_answer.count() # 11,858,865

# Clean tags, number of tags per post
drop_col = ('_PostTypeId', '_AcceptedAnswerId', '_CreationDate','_Body', 'AcceptedAnswerCreationDate')

df_question_with_answer = (df_question_with_answer
.drop(*drop_col)
.withColumn('_Tags', split('_Tags', '><'))
.selectExpr('*', "TRANSFORM(_Tags, value -> regexp_replace(value, '(>|<)', '')) AS tags_arr") # Clean tags, create array of tags
.withColumn('TagCountPerPost', size(array_distinct('tags_arr'))) # count number of tags per post
)

# df_questions.show()
df_question_with_answer.repartition(10).write.parquet('tag_question_with_answer.parquet') # Write parquet data to reuse later

# Explode by tag
df_question_with_answer = df_question_with_answer.withColumn('Tag', explode('tags_arr')).drop('_Tags').drop('tags_arr')

# Aggregation: top_tags table
top_tags_reponse = (
df_question_with_answer.groupBy('Tag') 
.agg(
    count('Tag').alias('tag_frequency'), # count tags
    F.expr('percentile(_Score, 0.5)').alias('med_score'), # median of score
    F.expr('percentile(_AnswerCount, 0.5)').alias('med_answer_count'), # median of answer count
    F.expr('percentile(AnswerDurationSeconds, 0.5)').alias('med_AnswerDurationSeconds'), # median of answer duration
    F.expr('percentile(AnswerDurationHour, 0.5)').alias('med_AnswerDurationHour'), # median of answer duration
    F.expr('percentile(AnswerDurationDays, 0.5)').alias('med_AnswerDurationDays'), # median of answer duration
    avg('AnswerDurationSeconds').alias('avg_AnswerDurationSeconds'),
    avg('AnswerDurationHour').alias('avg_AnswerDurationHour'),
    avg('AnswerDurationDays').alias('avg_AnswerDurationDays')
).orderBy(desc('tag_frequency'))
)
# top_tags_response.show()
top_tags_reponse.count()
top_tags_reponse.repartition(10).write.parquet('top_tags_response.parquet') # write parquet
