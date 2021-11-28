import pyspark, os
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,sum,avg
from time import time

corList = [1]
spark = SparkSession.builder.master("local[1]").appName("CS5488ProjectGroup5").getOrCreate()
df = spark.read.csv('../csv/pricechgdayhead.csv',header='true')
btcIndex = int(df.filter(df['Btc']==100000).collect()[0][50])
ethIndex = int(df.filter(df['Eth']==100000).collect()[0][50])
if btcIndex > ethIndex:
	listLen = ethIndex - 1
else:
	listLen = btcIndex - 1
coinDF = df.select(col('Btc').cast("double"),col('Eth').cast("double"),col('Index')).limit(listLen)
numPart = 51
partList = []
i = 1
while i < numPart:
	partList.append(i)
	i += 1
useTimeList = []
os.remove("timePartSpark.txt")
part = 1
while part < numPart:
	newDF = coinDF
	newDF = newDF.repartition(part)
	startTime = time()
	corVal = newDF.corr('Btc', 'Eth', "pearson")
	num = newDF.select('Eth').count()
	total = newDF.select(sum('Eth')).collect()[0][0]		
	average = newDF.select(avg('Eth')).collect()[0][0]								
	endTime = time()
	useTime = endTime - startTime
	useTimeList.append(useTime)
	f = open("timePartSpark.txt","a")
	f.write(str(part)+","+str(useTime)+"\n")
	f.close()
	part += 1
print(useTimeList)
plt.scatter(partList, useTimeList)
plt.title('Processing Time Vs Number of Partitions') 
plt.xlabel('Number of Partitions')
plt.ylabel('Processing Time (seconds)', rotation=90)
plt.show()
