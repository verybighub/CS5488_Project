import pyspark, os
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from time import time

spark = SparkSession.builder.appName('mlBtcEth').getOrCreate()
df = spark.read.csv('mlBtcEth.csv',header='true')   
df.toPandas()
numPart = 51
partList = []
i = 1
while i < numPart:
	partList.append(i)
	i += 1
useTimeList = []
os.remove("timePartML.txt")
part = 1
while part < numPart:
	newDF = df
	newDF = newDF.repartition(part)
	startTime = time()
	dataset=newDF.select(col('Btc').cast('float'),col('Eth').cast('float'),col('Move').cast('float'))
	required_features = ['Btc']
	assembler = VectorAssembler(inputCols=required_features, outputCol='features')
	transformed_data = assembler.transform(dataset)
	(training_data, test_data) = transformed_data.randomSplit([0.5,0.5])
	rf = RandomForestClassifier(labelCol='Move',featuresCol='features',maxDepth=5)
	model = rf.fit(training_data)
	predictions = model.transform(test_data)
	evaluator = MulticlassClassificationEvaluator(labelCol='Move',predictionCol='prediction',metricName='accuracy')
	accuracy = evaluator.evaluate(predictions)
	endTime = time()
	useTime = endTime - startTime
	useTimeList.append(useTime)
	f = open("timePartML.txt","a")
	f.write(str(part)+","+str(useTime)+"\n")
	f.close()
	part += 1
print(useTimeList)
plt.scatter(partList, useTimeList)
plt.title('Prediction Accuracy Processing Time Vs Number of Partitions') 
plt.xlabel('Number of Partitions')
plt.ylabel('Processing Time (seconds)', rotation=90)
plt.show()
