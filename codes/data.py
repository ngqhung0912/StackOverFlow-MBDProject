# -*- coding: utf-8 -*-
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode
import re
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
posts_df = posts_df.drop(*ignore_cols_posts)

CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
# accepted_answers = posts_df.join(questions.withColumnRenamed("_Id", "_PostId"), col("_Answer_Id") == col("_AcceptedAnswerId"))
accepted_answers = posts_df.filter(questions._AcceptedAnswerId == posts_df._Id)
#max min score
req = posts_df.groupBy("_Score")

distribution = req.count().sort("_Score", ascending=False).collect()
print("Min: ", distribution[-1])
print("Max: ", distribution[0])
'''
RESULTS
('Min: ', Row(_Score=1476, count=1))
('Max: ', Row(_Score=0, count=4620876))
('Mean: ', DataFrame[_Score: int, avg(_Score): double])
'''
#distribution
mean = req.avg("_Score")
print("Mean: ", mean)
#correlation