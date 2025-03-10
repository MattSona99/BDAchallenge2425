import os

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

entry_point = 'hdfs://user/amircoli/BDAchallenge2324'
save_point = 'hdfs://user/amircoli/Scrivania/BDA/spark2425/results/gruppo_1'

