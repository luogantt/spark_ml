#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:13:02 2018

@author: luogan
"""

from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors

from pyspark.sql import SparkSession
spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()
df = spark.createDataFrame([
    (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
    (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
    (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")

dctDf = dct.transform(df)

dctDf.select("featuresDCT").show(truncate=False)