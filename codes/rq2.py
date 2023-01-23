# ==================
# MBD Project RQ 2
# ==================

# Script to run:
# time spark-submit --master yarn --deploy-mode cluster --conf spark.dynamicAllocation.maxExecutors=20 rq2.py 2> /dev/null

# Copy parquet to local:
# STEP 1: hdfs dfs -copyToLocal /user/s2767708/project/Q2/rq2_3_1m.parquet
# STEP 2: scp -r s2767708@ctit008.ewi.utwente.nl:/home/s2767708/rq2_3_1m.parquet /Users/HP/Desktop

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from datetime import datetime

sc = SparkContext(appName="Project_Q2")
spark = SparkSession.builder.getOrCreate()
sc.setLogLevel("ERROR")

# Read datasets
badges = spark.read.parquet('/user/s2812940/project/parquet_data/badges.parquet')
posts = spark.read.parquet('/user/s2812940/project/parquet_data/posts.parquet')
users = spark.read.parquet('/user/s2812940/project/parquet_data/users.parquet')

post_id = posts.select(col('_Id'), col('_OwnerUserId'))
user_rep = users.select(col('_Id'), col('_Reputation'))



# =================================================
# RQ 2.1 Are user reputation correlated to badges?
# =================================================

## Group badges on users
badges_gsb = badges.select(col('_UserId')).groupBy(col('_UserId')).agg(count("*").alias('count_all_badges'))\
    .join(badges.select(col('_UserId'), col('_Class')).filter(col('_Class')==1).groupBy(col('_UserId')).agg(count('*').alias('count_gold')), ['_UserId'], 'left')\
    .join(badges.select(col('_UserId'), col('_Class')).filter(col('_Class')==2).groupBy(col('_UserId')).agg(count('*').alias('count_silver')), ['_UserId'], 'left')\
    .join(badges.select(col('_UserId'), col('_Class')).filter(col('_Class')==3).groupBy(col('_UserId')).agg(count('*').alias('count_bronze')), ['_UserId'], 'left')

# Join badges_gsb and users tables
users_badges = users.select(col('_Id'), col('_Reputation'))\
    .join(badges_gsb, [users._Id == badges_gsb._UserId], 'left')\
    .na.fill(0)\
    .drop('_UserId')

# Save result
users_badges.repartition(100).write.parquet('/user/s2767708/project/Q2/rq2_1.parquet')



# ============================================================
# RQ 2.2 Is user reputation correlated to quality of answer?
# ============================================================

## Metric 1: Sum of score
## ------------------------
answer_score = posts.filter(col('_PostTypeId')==2)\
    .select(col('_Score'), col('_OwnerUserId'))\
    .groupBy(col('_OwnerUserId'))\
    .agg(sum(col('_Score')).alias('sum_score'))

rq2_2_1 = users.select(col('_Id'), col('_Reputation'))\
    .join(answer_score, [users._Id == answer_score._OwnerUserId], 'left')\
    .na.fill(0)\
    .drop('_OwnerUserId')

# Save result
rq2_2_1.write.parquet('/user/s2767708/project/Q2/rq2_2_1.parquet')



## Metric 1: Average of score
## ----------------------------
question_score = posts.filter(col('_PostTypeId')==1)\
    .select(col('_Score'), col('_OwnerUserId'))\
    .groupBy(col('_OwnerUserId'))\
    .agg(mean(col('_Score')).alias('avg_score'))

rq2_2_1m = users.select(col('_Id'), col('_Reputation'))\
    .join(question_score, [users._Id == question_score._OwnerUserId], 'left')\
    .na.fill(0)\
    .drop('_OwnerUserId')

# Save result
rq2_2_1m.write.parquet('/user/s2767708/project/Q2/rq2_2_1m.parquet')



## Metric 2: Count of accepted answers
## -------------------------------------
acpt_answer = posts.select(col('_AcceptedAnswerId'))\
    .filter(col('_AcceptedAnswerId').isNull()==False)
    
acpt_answer_id = acpt_answer.join(post_id, [acpt_answer._AcceptedAnswerId == post_id._Id], 'left')\
    .drop('_AcceptedAnswerId', '_Id')\
    .filter(col('_OwnerUserId').isNull()==False)

acpt_answer_group = acpt_answer_id.groupBy(col('_OwnerUserId').alias('_Id'))\
    .agg(count('*').alias('count_accepted_answer'))

rq2_2_2 = acpt_answer_group.join(user_rep, ['_Id'], 'left')

# Save result
rq2_2_2.write.parquet('/user/s2767708/project/Q2/rq2_2_2.parquet')



# ==============================================================
# RQ 2.3 Is user reputation correlated to quality of question?
# ==============================================================

# Metric 1: Sum of score
# ------------------------
question_score = posts.filter(col('_PostTypeId')==1)\
    .select(col('_Score'), col('_OwnerUserId'))\
    .groupBy(col('_OwnerUserId'))\
    .agg(sum(col('_Score')).alias('sum_score'))

rq2_3_1 = users.select(col('_Id'), col('_Reputation'))\
    .join(question_score, [users._Id == question_score._OwnerUserId], 'left')\
    .na.fill(0)\
    .drop('_OwnerUserId')

# Save result
rq2_3_1.write.parquet('/user/s2767708/project/Q2/rq2_3_1.parquet')



## Metric 1: Average of score
## ----------------------------
question_score = posts.filter(col('_PostTypeId')==1)\
    .select(col('_Score'), col('_OwnerUserId'))\
    .groupBy(col('_OwnerUserId'))\
    .agg(mean(col('_Score')).alias('avg_score'))

rq2_3_1m = users.select(col('_Id'), col('_Reputation'))\
    .join(question_score, [users._Id == question_score._OwnerUserId], 'left')\
    .na.fill(0)\
    .drop('_OwnerUserId')

# Save result
rq2_3_1m.write.parquet('/user/s2767708/project/Q2/rq2_3_1m.parquet')



# Metric 2: Sum of view counts
# ------------------------------
question_viewcount = posts.filter(col('_PostTypeId')==1)\
    .select(col('_ViewCount'), col('_OwnerUserId'))\
    .groupBy(col('_OwnerUserId'))\
    .agg(sum(col('_ViewCount')).alias('sum_viewcount'))
    
rq2_3_2 = users.select(col('_Id'), col('_Reputation'))\
    .join(question_viewcount, [users._Id == question_viewcount._OwnerUserId], 'left')\
    .na.fill(0)\
    .drop('_OwnerUserId')

# Save result
rq2_3_2.write.parquet('/user/s2767708/project/Q2/rq2_3_2.parquet')



# Metric 3: Percentage of answered questions
# --------------------------------------------
question_asked = posts.filter(col('_PostTypeId')==1)\
    .select(col('_OwnerUserId'))\
    .groupBy(col('_OwnerUserId'))\
    .agg(count('*').alias('question_asked'))
    
question_answered = posts.filter((col('_PostTypeId')==1) & (col('_AcceptedAnswerId').isNull()==False))\
    .select(col('_OwnerUserId'))\
    .groupBy(col('_OwnerUserId'))\
    .agg(count('*').alias('question_answered'))
    
user_rep = users.select(col('_Id'), col('_Reputation'))

question_aa = question_asked.join(question_answered, ['_OwnerUserId'], 'left')\
    .na.fill(0)\
    .withColumn('percent_answered', round(col('question_answered')/col('question_asked'), 4))
    
rq2_3_3 = question_aa.join(user_rep, [question_aa._OwnerUserId == user_rep._Id], 'left')\
    .drop('_Id')
    
# Save result
rq2_3_3.write.parquet('/user/s2767708/project/Q2/rq2_3_3.parquet')
