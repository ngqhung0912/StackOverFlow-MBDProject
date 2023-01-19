from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

METRICS_PATH = "/user/s2812940/project/parquet_data/metrics.parquet"
spark = SparkSession.builder.appName("Colyn").getOrCreate()

dataset = spark.read.parquet(METRICS_PATH)

# Next, in order to train ML models in Spark later, we'll use the VectorAssembler to combine a given list of columns into a single vector column.
assembler = VectorAssembler(inputCols='_Score', outputCols='features')
df = assembler.transform(dataset).select('features')
df.show(6)

# Next, we standardize the features, notice here we only need to specify the assembled column as the input feature
scaler = StandardScaler(
    inputCol='features',
    outputCol='scaledFeatures',
    withMean=True,
    withStd=True
).fit(df)

df_scaled = scaler.transform(df)
df_scaled.show(6)

# After the preprocessing step, we fit the PCA model.
n_components = 2
pca = PCA(
    k=n_components,
    inputCol='scaledFeatures',
    outputCol='pcaFeatures'
).fit(df_scaled)

df_pca = pca.transform(df_scaled)
print('Explained variance Ratio', pca.explainedVariance.toArray())
df_pca.show(6)


