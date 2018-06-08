#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:26:48 2018

@author: luogan
"""

from pyspark.ml.feature import VectorIndexer

from pyspark.sql import SparkSession

spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()

data = spark.read.format("libsvm").load("/home/luogan/lg/softinstall/spark-2.2.0-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")

indexer = VectorIndexer(inputCol="features", outputCol="indexed", maxCategories=10)
indexerModel = indexer.fit(data)

categoricalFeatures = indexerModel.categoryMaps
print("Chose %d categorical features: %s" %
      (len(categoricalFeatures), ", ".join(str(k) for k in categoricalFeatures.keys())))

# Create new column "indexed" with categorical values transformed to indices
indexedData = indexerModel.transform(data)
indexedData.show()