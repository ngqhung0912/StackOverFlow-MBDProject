'''
Research question 1: Pre-process tags for trend analysis
This script is use to pre-process tags based on year. Output will be used to visualize trend analysis

Run this script in the cluster using following command:
time spark-submit --master yarn \
    --deploy-mode cluster \
    --conf spark.dynamicAllocation.maxExecutors=15 \
    --conf spark.dynamicAllocation.minExecutors=10 \
    --conf spark.dynamicAllocation.initialExecutors=5 \
    --jars /home/s2812940/spark-xml_2.11-0.9.0.jar top_tags_year.py 2> /dev/null
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

# Read pre-processed tags data
tags_distribution = spark.read.parquet('/user/s2812940/tags_distribution.parquet')

# Aggregation: top_tags_year table
top_tags_year = (
tags_distribution.groupBy('Tag', 'Year') 
.agg(
    count('Tag').alias('tag_frequency'), # count tags
    F.expr('percentile(_Score, 0.5)').alias('med_score'), # median of score
    F.expr('percentile(_AnswerCount, 0.5)').alias('med_answer_count'), # median of answer count
    (sum('is_answer')/count('Tag')).alias('percent_answer') # percentage of getting answer
).orderBy(desc('tag_frequency'))
)
# top_tags.show()
top_tags_year.count()
top_tags_year.repartition(10).write.parquet('top_tags_year.parquet') # write parquet

