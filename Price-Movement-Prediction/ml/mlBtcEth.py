from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

trainPortion = [0.5,0.6,0.7,0.8,0.9,0.95]
spark = SparkSession.builder.appName('mlBtcEth').getOrCreate()
df = spark.read.csv('mlBtcEth.csv',header='true')   
df.toPandas()
dataset=df.select(col('Btc').cast('float'),col('Eth').cast('float'),col('Move').cast('float'))
required_features = ['Btc']
assembler = VectorAssembler(inputCols=required_features, outputCol='features')
transformed_data = assembler.transform(dataset)
for i in trainPortion:
	(training_data, test_data) = transformed_data.randomSplit([i,1-i])
	rf = RandomForestClassifier(labelCol='Move',featuresCol='features',maxDepth=5)
	model = rf.fit(training_data)
	predictions = model.transform(test_data)
	evaluator = MulticlassClassificationEvaluator(labelCol='Move',predictionCol='prediction',metricName='accuracy')
	accuracy = evaluator.evaluate(predictions)
	print('Training portion: '+str(i)+', Test Accuracy: '+str(accuracy))
