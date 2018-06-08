#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:36:15 2018

@author: luogan
"""

from pyspark.ml.clustering import GaussianMixture
from pyspark.sql import SparkSession
spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()
# loads data
dataset = spark.read.format("libsvm").load("/home/luogan/lg/softinstall/spark-2.3.0-bin-hadoop2.7/data/mllib/sample_kmeans_data.txt")

gmm = GaussianMixture().setK(2).setSeed(538009335)
model = gmm.fit(dataset)

print("Gaussians shown as a DataFrame: ")
model.gaussiansDF.show(truncate=False)
