#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 16:27:57 2018

@author: luogan
"""

import pandas as pd
from pyspark.sql import SparkSession
spark= SparkSession\
                .builder \
                .appName("dataFrame") \
                .getOrCreate()
# Loads data.


ll3=pd.DataFrame([[1,2],[3,4]],columns=['a','b'])



cc=ll3.values.tolist()

dd=list(ll3.columns)
#df=spark.createDataFrame(ll3)

#turn pandas.DataFrame  to spark.dataFrame
spark_df  = spark.createDataFrame(cc, dd)

print('spark.dataFram=',spark_df.show())

#turn spark.dataFrame to pandas.DataFrame  
pandas_df = spark_df .toPandas()  

print('pandas.DataFrame=',pandas_df)