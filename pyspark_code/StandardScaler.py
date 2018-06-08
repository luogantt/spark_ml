#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:29:37 2018

@author: luogan
"""

from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession

spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()
dataFrame = spark.read.format("libsvm").load("/home/luogan/lg/softinstall/spark-2.2.0-bin-hadoop2.7/data/mllib/sample_libsvm_data.txt")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(dataFrame)

# Normalize each feature to have unit standard deviation.
scaledData = scalerModel.transform(dataFrame)
scaledData.show()