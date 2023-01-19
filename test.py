# -*- coding: utf-8 -*-
# /home/s2096307/miniconda3/envs/bigdataEnv.zip
from codes.features import word_count, sentence_count, char_count, coleman_liau, clean_data

# import nltk
# from nltk.sentiment import SentimentAnalyzer
#
# nltk.download('punkt')
from pyspark.sql import SparkSession

from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
import pyspark.sql.functions as f

from pyspark.sql.functions import udf, col

'''
PYSPARK_PYTHON=./../miniconda3/envs/bigdataEnv/bin/python spark-submit  --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./../miniconda3/envs/bigdataEnv/bin/python --archives /../miniconda3/envs/bigdataEnv.zip#bigdataEnv test.py

'''
def test(text):
    text = clean_data(text)
    print(word_count(text))
    print(char_count(text))
    print(sentence_count(text))
    print(coleman_liau(char_count(text), sentence_count(text), word_count(text)))
    print(text)


spark = SparkSession.builder.appName("Colyn").getOrCreate()
small_data_path = "posts.parquet/part-00000-dfdedfcd-0d15-452e-bab4-48f6cf9a8276-c000.snappy.parquet"
posts_df = spark.read.parquet(small_data_path)

test("<p>(can't fit in comment , put it here)</p><p>I think , there is pretty much nothing you can do about this , depending on your hardware , 11 min  can be actually not bad , in the execution plan , I can see everything looks ok.</p> \
<p>but for your information , bottleneck in that insert statement is to read data from T&quot;Recovery..tags&quot; table, which took 07 minutes of your query time.(It's used full scan which is ok considering it needs to read 2 million rows and return a lot of columns)</p>\
<p>so the only thing you can do is to find a way to speed up reading from linked server &quot;Recovery&quot;.\
linked servers are usually the source of poor performance, specially huge data ,which can be due to poor network or busy network , etc ...</p>\
<p>anyways one solution is :</p>\
<ul>\
<li>pull data from linked server into a table into R3 server ( directly) with server in middle.which depends on your scenario you can</li>\
<li>change your query to be against that table</li>\
</ul>\
<p>this can significantly improve your query time</p> <pre><code>template &lt;class T&gt;\
void Lista&lt;T&gt;::imprimir()\
{\
    NodoL *ptr = new NodoL;\
    ptr-&gt;sig = pri-&gt;sig;\
    cout &lt;&lt; *ptr-&gt;sig-&gt;elem; //THIS DISPLAYS CORRECTLY\
    if(ptr-&gt;sig == NULL || ptr-&gt;sig-&gt;sig == NULL)\
       return;\
    cout &lt;&lt; *ptr-&gt;sig-&gt;sig-&gt;elem; //SEGMENTATION FAULT\
}\
</code></pre><code>trn;\
    cout &lt;&lt; *ptr-&gt;sig-&gt;sig-&gt;elem; //SEGMENTATION FAULT\
}\
</code></pre>")
post_df_small = posts_df.limit(1)
word_count_udf = udf(lambda x: word_count(x))
post_df_small = post_df_small.withColumn('bodyWordCount', word_count_udf(col('_Body')))
print(post_df_small.show())
