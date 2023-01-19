from codes.features import Features

import nltk
nltk.download('punkt')

from pyspark import SparkContext
from pyspark.sql import SQLContext
import sys
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pyspark.sql.functions as f

from pyspark.sql.functions import udf, col

spark =

def test(text):
    features = Features(text)
    print(features.coleman_liau())
    print(features.flesch_ease())
    print(features.flesch_grade())
    print(features.get_content())


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

# word_count_udf = udf(lambda x: word_count(x))
# post_df_small = post_df_small.withColumn('bodyWordCount', word_count_udf(col('_Body')))
# print(post_df_small.show())