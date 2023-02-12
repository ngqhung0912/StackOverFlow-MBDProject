'''
Research question 1: Analysing tags and posts
This script used to filter tags based on cumulative frequency (80%, 85% and 90%).
Output are extracted to csv file.

Run this script in the cluster using following command:
time spark-submit --master yarn \
    --deploy-mode cluster \
    --conf spark.dynamicAllocation.maxExecutors=15 \
    --conf spark.dynamicAllocation.minExecutors=10 \
    --conf spark.dynamicAllocation.initialExecutors=5 \
    --jars /home/s2812940/spark-xml_2.11-0.9.0.jar top_tags_final.py 2> /dev/null
'''
from pyspark import SparkContext
from pyspark.sql import SQLContext

import sys
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.sql.window import Window

#sc.stop()
sc = SparkContext(appName="stack_exchange")
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()

# Read data 
top_tags = spark.read.parquet('/user/s2812940/top_tags.parquet')
top_tags.count()
top_tags.show()

# Create a column for relative frequency
top_tags = (top_tags
.withColumn('rel_frequency', F.col('tag_frequency')/F.sum('tag_frequency').over(Window.partitionBy()))
).orderBy(desc('rel_frequency'))

# Create a column for cumulative frequency
top_tags = (top_tags
.withColumn('cum_frequency',
expr('sum(rel_frequency) over(order by tag_frequency DESC)'))
)

top_tags.show()

# Filter 80% of tags 
top_tags_80 = top_tags.filter(col('cum_frequency') <= 0.80)
top_tags_80.count() # 1680
top_tags_80.write.option("header","true").csv('top_tags_80.csv')

# Filter 85% of tags
top_tags.filter(col('cum_frequency') <= 0.85).count() # 2728
top_tags_80.write.option("header","true").csv('top_tags_85.csv')

# Filter 90% of tags 
top_tags.filter(col('cum_frequency') <= 0.90).count() #4748
top_tags_80.write.option("header","true").csv('top_tags_90.csv')