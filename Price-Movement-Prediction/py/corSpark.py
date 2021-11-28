import pyspark
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from coinList import coinList1


coinList = coinList1()
corList = [1]
os.remove("corResult.txt")
for i in coinList:
	if i != "Btc":
		spark = SparkSession.builder.master("local[1]").appName("CS5488ProjectGroup5").getOrCreate()
		df = spark.read.csv('../csv/pricechgdayhead.csv',header='true')
		baseIndex = int(df.filter(df[coinList[0]]==100000).collect()[0][50])
		altIndex = int(df.filter(df[i]==100000).collect()[0][50])
		if baseIndex > altIndex:
			listLen = altIndex - 1
		else:
			listLen = baseIndex - 1
		coinDF = df.select(col('Btc').cast("double"),col(i).cast("double"),col('Index')).limit(listLen)
		corVal = coinDF.corr('Btc', i, "pearson")
		corList.append(corVal)
		f = open("corResult.txt","a")
		f.write(str(i)+","+str(corVal)+"\n")
		f.close()
plt.scatter(coinList, corList)
plt.title('Correlation between Bitcoin and Altcoins') 
plt.xlabel('Altcoins')
plt.ylabel('Correlation Coefficient', rotation=90)
plt.xticks(rotation=90, size=5)
plt.show()	
