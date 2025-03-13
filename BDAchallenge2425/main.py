import os
import json
import threading
from time import time

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, count, size, input_file_name, lit, row_number, regexp_extract, expr, avg, explode
from pyspark.sql.types import FloatType

from pyspark.sql import Window

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)
start_time = time()

HDFS_PATH = 'hdfs:///user/user/dataset'
SAVING_PATH = os.path.abspath('output')
COLUMNS = ['LATITUDE', 'LONGITUDE', 'WND', 'TMP', 'REM']

try:
    os.makedirs(SAVING_PATH)
except OSError:
    pass

def get_file_paths(path):
    """
    Restituisce un elenco dei nomi dei file presenti nel percorso specificato su HDFS.
    """
    return [files.getPath().getName() 
            for files in spark._jvm.org.apache.hadoop.fs.FileSystem
            .get(spark._jsc.hadoopConfiguration())
            .listStatus(spark._jvm.org.apache.hadoop.fs.Path(path))]

def get_headers():
    """
    Legge gli header dai file CSV in HDFS e restituisce un dizionario.
    La chiave è una tupla con gli indici delle colonne di interesse,
    il valore è una lista di percorsi dei file che hanno lo stesso schema di colonne.
    """
    headers = {}
    for year in get_file_paths(HDFS_PATH):
        for station in get_file_paths('{}/{}'.format(HDFS_PATH, year)):
            path = '{}/{}/{}'.format(HDFS_PATH, year, station)
            header = spark.read.option("header", "true").csv(path).columns
            # Another way to go is with smart_open:
            # header = smart_open.smart_open('{}/{}/{}'.format(dataset_path, year, station), 'r').readline().strip()
            key = tuple(header.index(c) for c in COLUMNS)
            headers[key] = headers.get(key, []) + [(year, station)]
    return headers

def read_csv(files=None):
    """
    Legge i file CSV specificati e restituisce un DataFrame Spark.
    """
    if files is None:
        files = '{}//.csv'.format(HDFS_PATH)
    
    return spark.read.format('csv') \
        .option('header', 'true') \
        .option('enforceSchema', 'false') \
        .load(files) \
        .withColumn('file_path', input_file_name()) \
        .withColumn('year', split(col('file_path'), '/')[size(split(col('file_path'), '/')) - 2]) \
        .withColumn('station', split(col('file_path'), '/')[size(split(col('file_path'), '/')) - 1].substr(1, 11))

def write_to_file(filename, df):
    """
    Scrive un DataFrame in un file CSV nella cartella di output.
    """
    df.write \
        .format("csv") \
        .option("header", "true") \
        .mode("overwrite") \
        .save('file://{}/{}'.format(SAVING_PATH, filename))

def task1(df):
    """
    Analizza la temperatura media nelle coordinate specificate,
    conta le occorrenze e salva i primi 10 risultati.
    """
    start = time()
    result_df = df \
        .withColumn("LATITUDE", col('LATITUDE').cast(FloatType())) \
        .withColumn("LONGITUDE", col('LONGITUDE').cast(FloatType())) \
        .withColumn("TMP", split(col('TMP'), ',')[0].cast(FloatType()) / 10) \
        .select(['LATITUDE', 'LONGITUDE', 'TMP']) \
        .filter((col('LATITUDE') >= 30) & (col('LATITUDE') <= 60) &
                (col('LONGITUDE') >= -135) & (col('LONGITUDE') <= -90)) \
        .groupBy('TMP') \
        .agg(count('*').alias('num_occurrences')) \
        .orderBy(col("num_occurrences").desc(), col("TMP").asc()) \
        .withColumn('Location', lit('[(60,-135);(30,-90)]')) \
        .select(['Location', 'TMP', 'num_occurrences']) \
        .limit(10)
    write_to_file('task1', result_df)
    print("tempo 1 {} s.".format(time() - start_time))
    return time() - start

def task2(df):
    """
    Analizza i dati del vento (WND) e restituisce la stazione con il numero maggiore di occorrenze per ogni valore di WND.
    """
    start = time()
    result_df = df \
        .select(['station', 'WND']) \
        .withColumn('WND', split(col('WND'), ',')[1]) \
        .groupBy('WND', 'station') \
        .agg(count('*').alias('num_occurrences')) \
        .orderBy(col("num_occurrences").desc(), col('station').asc(), col("WND").asc()) \
        .withColumn("rank", row_number().over(Window.partitionBy("WND").orderBy(col("num_occurrences").desc(), col("station").asc()))) \
        .filter(col("rank") == 1) \
        .drop("rank") \
        .orderBy(col("WND").asc())
    write_to_file('task2', result_df)
    print("tempo 2 {} s.".format(time() - start_time))
    return time() - start

def task3(df):
    """
    Analizza le precipitazioni medie annuali per ogni stazione e restituisce le 10 con il valore più basso.
    """
    start = time()
    
    result_df = df \
        .select(['year', 'station', 'REM']) \
        .filter(col('REM').isNotNull()) \
        .filter(col('REM').contains('HOURLY INCREMENTAL PRECIPITATION VALUES (IN)')) \
        .withColumn('precipitation_values', 
                   regexp_extract(col('REM'), 
                                 'HOURLY INCREMENTAL PRECIPITATION VALUES \\(IN\\):([^,]*)', 1)) \
        .withColumn('precipitation_values', 
                   expr("regexp_replace(precipitation_values, 'T', '0')")) \
        .withColumn('precipitation_array', 
                   split(col('precipitation_values'), ' ')) \
        .withColumn('precipitation_array', 
                   expr("transform(precipitation_array, x -> case when x = '' then null when x = 'T' then '0' else x end)")) \
        .withColumn('precipitation_array', 
                   expr("filter(precipitation_array, x -> x is not null)")) \
        .withColumn('precipitation_value', 
                   explode(col('precipitation_array'))) \
        .withColumn('precipitation_value', 
                   col('precipitation_value').cast(FloatType())) \
        .filter(col('precipitation_value').isNotNull()) \
        .groupBy('year', 'station') \
        .agg(avg('precipitation_value').alias('avg_precipitation')) \
        .withColumn("rank", row_number().over(Window.partitionBy("year").orderBy(col("avg_precipitation").asc(), col("station").asc()))) \
        .filter(col("rank") <= 10) \
        .drop("rank") \
        .orderBy(col("year").asc(), col("station").asc()).limit(10)
    
    write_to_file('task3', result_df)
    print("tempo 3 {} s.".format(time() - start_time))
    return time() - start

def run_tasks_in_threads(df):
    """
    Esegue le tre task in parallelo utilizzando thread separati.
    """
    thread1 = threading.Thread(target=task1, args=(df,))
    thread2 = threading.Thread(target=task2, args=(df,))
    thread3 = threading.Thread(target=task3, args=(df,))

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()


if __name__ == '__main__':
    headers = get_headers()
    dfs = []
    for stations in headers.values():
        files = ['{}/{}/{}'.format(HDFS_PATH, year, station) for year, station in stations]
        dfs.append(read_csv(files))
    union_df = dfs[0]
    for df in dfs[1:]:
        union_df = union_df.unionByName(df, allowMissingColumns=True)

    run_tasks_in_threads(union_df)

    print(f"total time {time() - start_time} seconds.")