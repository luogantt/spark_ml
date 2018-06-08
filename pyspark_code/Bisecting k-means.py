#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 11:33:35 2018

@author: luogan
"""

from pyspark.ml.clustering import BisectingKMeans

from pyspark.sql import SparkSession
spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()
# Loads data.
dataset = spark.read.format("libsvm").load("/home/luogan/lg/softinstall/spark-2.3.0-bin-hadoop2.7/data/mllib/sample_kmeans_data.txt")


# Trains a bisecting k-means model.
bkm = BisectingKMeans().setK(2).setSeed(1)
model = bkm.fit(dataset)

# Evaluate clustering.
cost = model.computeCost(dataset)
print("Within Set Sum of Squared Errors = " + str(cost))

# Shows the result.
print("Cluster Centers: ")
centers = model.clusterCenters()
for center in centers:
    print(center)