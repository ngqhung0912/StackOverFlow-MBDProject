'''
This script is used to import original data, then convert to parquet format.
Update: 09/01/23 (Sam)

Disclaimer: 
Dependency: spark-xml packages (spark-xml_2.11-0.9.0.jar)
Parquet files are available at: /user/s2812940/project/parquet_data
Posts and Users tables take really long time to write parquet files
----
Pipeline: 
1. Defind schema of table. Note: all the fields with datetime type are defined as string to avoid null data issue.
They should be converted back to datetime after importing data.
2. Loading xml.gz big data file, using pre-defined schema.
3. Repartition dataframe (10 partitions) and write output as parquet.
---
Run this script in the cluster: (Must load the spark-xml packages by --jars)

time spark-submit --master yarn \
--deploy-mode cluster \
--conf spark.dynamicAllocation.minExecutors=10 \
--conf spark.dynamicAllocation.maxExecutors=15 \
--jars hdfs:/user/s2812940/project/spark-xml_2.11-0.9.0.jar create_parquet.py 2> /dev/null
'''

from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
from pyspark.sql.types import *
from pyspark.sql import SparkSession

sc = SparkContext(appName="stack_exchange")
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)
spark = SparkSession.builder.getOrCreate()

# Table tags. Note: All datetime fields need to be converted from string to timestamp
schema_tags = StructType([
    StructField('_Id', IntegerType()),
    StructField('_TagName', StringType()),
    StructField('_Count', IntegerType()),
    StructField('_ExcerptPostId', IntegerType()),
    StructField('_WikiPostId', IntegerType())
		])
filename = '/user/s2812940/project/data/Tags.xml.gz'
tags = spark.read.format('com.databricks.spark.xml') \
    .options(rowTag = 'row') \
    .options(numPartitions = '20') \
    .schema(schema_tags) \
    .load(filename)

# repartition the dataframe to 10 partitions, save data to parquet format to import faster
tags.repartition(10).write.parquet("tags.parquet") 
# We can check the parquet data with this code: spark.read.parquet('/user/s2812940/tags10.parquet')

# Table postlinks. Note: All datetime fields need to be converted from string to timestamp
schema_postlinks = StructType([
    StructField('_Id', IntegerType()),
    StructField('_CreationDate', StringType()), # Should change this to timestamp later
    StructField('_PostId', IntegerType()),
    StructField('_RelatedPostId', IntegerType()),
    StructField('_LinkTypeId', IntegerType())
		])

filename = '/user/s2812940/project/data/PostLinks.xml.gz'

postlinks = spark.read.format('com.databricks.spark.xml') \
    .options(rowTag = 'row') \
    .options(numPartitions = '20') \
    .schema(schema_postlinks) \
    .load(filename)

postlinks.repartition(10).write.parquet("postlinks.parquet") 

# Table badges. Note: All datetime fields need to be converted from string to timestamp
schema_badges = StructType([
    StructField('_Id', IntegerType()),
    StructField('_UserId', IntegerType()),
    StructField('_Name', StringType()),
    StructField('_Date', StringType()), # Should change this to timestamp later
    StructField('_Class', IntegerType()),
    StructField('_TagBased', BooleanType())
		])
filename = '/user/s2812940/project/data/Badges.xml.gz'

badges = spark.read.format('com.databricks.spark.xml') \
    .options(rowTag = 'row') \
    .options(numPartitions = '20') \
    .schema(schema_badges) \
    .load(filename)

badges.repartition(10).write.parquet("badges.parquet")

# Table users and posts take long time for writing parquets.

# Table users. Note: All datetime fields need to be converted from string to timestamp
schema_users = StructType([
    StructField('_Id', IntegerType()),
    StructField('_Reputation', IntegerType()),
    StructField('_CreationDate', StringType()),  # Should change this to timestamp later
    StructField('_DisplayName', StringType()),
	StructField('_LastAccessDate', StringType()),  # Should change this to timestamp later
	StructField('_AboutMe', StringType()),
	StructField('_Views', IntegerType()),
	StructField('_UpVotes', IntegerType()),
	StructField('_DownVotes', IntegerType())
])

filename = '/user/s2812940/project/data/Users.xml.gz'

users = spark.read.format('com.databricks.spark.xml') \
    .options(rowTag = 'row') \
    .options(numPartitions = '20') \
    .schema(schema_users) \
    .load(filename)

users.repartition(10).write.parquet("users.parquet")

# Table Posts. Note: All datetime fields need to be converted from string to timestamp
schema_posts = StructType([
    StructField('_Id', IntegerType()),
    StructField('_PostTypeId', IntegerType()),
    StructField('_AcceptedAnswerId', IntegerType()),
    StructField('_CreationDate', StringType()), # Should change this to timestamp later
    StructField('_Score', IntegerType()),
    StructField('_ViewCount', IntegerType()),
    StructField('_Body', StringType()),
    StructField('_OwnerUserId', IntegerType()),
    StructField('_LastEditorUserId', StringType()),
    StructField('_LastEditorDisplayName', StringType()),
    StructField('_LastEditDate', StringType()), # Should change this to timestamp later
    StructField('_LastActivityDate', StringType()), # Should change this to timestamp later
    StructField('_Title', StringType()),
    StructField('_Tags', StringType()),
    StructField('_AnswerCount', IntegerType()),
    StructField('_CommentCount', IntegerType()),
    StructField('_FavoriteCount', IntegerType()),
    StructField('_CommunityOwnedDate', StringType()), # Should change this to timestamp later
    StructField('_ContentLicense', StringType())
])

filename = '/user/s2812940/project/data/Posts.xml.gz'

posts = spark.read.format('com.databricks.spark.xml') \
    .options(rowTag = 'row') \
    .options(numPartitions = '20') \
    .schema(schema_posts) \
    .load(filename)

posts.repartition(10).write.parquet("posts.parquet")