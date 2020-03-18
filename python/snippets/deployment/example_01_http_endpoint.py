## Copyright 2019 The SQLNet Company GmbH

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import json
from urllib import (
    error,
    request
)

import numpy as np
import pandas as pd

import getml.models.aggregations as aggregations
import getml.datasets as datasets
import getml.engine as engine
import getml.models.loss_functions as loss_functions
import getml.models as models
import getml.data as data
import getml.predictors as predictors

# ----------------

engine.set_project("examples")

# ----------------
# Generate artificial dataset
# The problem we create looks like this:
#
# SELECT COUNT( * )
# FROM POPULATION t1
# LEFT JOIN PERIPHERAL t2
# ON t1.join_key = t2.join_key
# WHERE (
#    ( t1.time_stamp - t2.time_stamp <= 0.5 )
# ) AND t2.time_stamp <= t1.time_stamp
# GROUP BY t1.join_key,
#          t1.time_stamp;

population_table, peripheral_table = datasets.make_numerical()

population_placeholder = population_table.to_placeholder()
peripheral_placeholder = peripheral_table.to_placeholder()
population_placeholder.join(peripheral_placeholder, "join_key", "time_stamp")

predictor = predictors.LinearRegression()

model = models.MultirelModel(
    name="MyModel",
    aggregation=[
        aggregations.Count,
        aggregations.Sum
    ],
    population=population_placeholder,
    peripheral=[peripheral_placeholder],
    loss_function=loss_functions.SquareLoss(),
    predictor=predictor,
    num_features=10,
    share_aggregations=1.0,
    max_length=1,
    num_threads=0
).send()

# ----------------

model = model.fit(
    population_table=population_table,
    peripheral_tables=[peripheral_table]
)

# ----------------

features = model.transform(
    population_table=population_table,
    peripheral_tables=[peripheral_table]
)

# ----------------

yhat = model.predict(
    population_table=population_table,
    peripheral_tables=[peripheral_table]
)

# ----------------

print(model.to_sql())

# ----------------
# By the way, passing pandas.DataFrames still works.

scores = model.score(
    population_table=population_table,
    peripheral_tables=[peripheral_table]
)

print(scores)

# ----------------
# Before you can send HTTP requests to the model,
# you need to activate that.

model.deploy(True)

# ----------------
# In order to make the model generate predictions,
# you can simply send a POST request:

data = """{
  "peripheral": [{
    "column_01": [2.4, 3.0, 1.2, 1.4, 2.2],
    "join_key": ["0", "0", "0", "0", "0"],
    "time_stamp": [0.1, 0.2, 0.3, 0.4, 0.8]
  }],
  "population": {
    "column_01": [2.2, 3.2],
    "join_key": ["0", "0"],
    "time_stamp": [0.65, 0.81]
  }
}"""

url = "http://localhost:1709/predict/MyModel/"

try:
    response = request.urlopen(
      url=url, 
      data=data.encode()
    ).read()
except error.HTTPError as e:
    error_message = e.read()
    raise Exception(error_message)

print(response)

# ----------------
# This is how you get the features.

url = "http://localhost:1709/transform/MyModel/"

try:
    response = request.urlopen(
      url=url, 
      data=data.encode()
    ).read()
except error.HTTPError as e:
    error_message = e.read()
    raise Exception(error_message)

print(response)

# ---------------
# Using time stamps 

data2 = """{
		"peripheral": [{
			"column_01": [2.4, 3.0, 1.2, 1.4, 2.2],
			"join_key": ["0", "0", "0", "0", "0"],
			"time_stamp": ["2010-01-01 00:15:00", "2010-01-01 08:00:00", "2010-01-01 09:30:00", "2010-01-01 13:00:00", "2010-01-01 23:35:00"]
		}],
		"population": {
			"column_01": [2.2, 3.2],
			"join_key": ["0", "0"],
			"time_stamp": ["2010-01-01 12:30:00", "2010-01-01 23:30:00"]
		},
		"timeFormats": ["%Y-%m-%d %H:%M:%S"]
	}"""

url = "http://localhost:1709/predict/MyModel/"

try:
    response = request.urlopen(
      url=url, 
      data=data.encode()
    ).read()
except error.HTTPError as e:
    error_message = e.read()
    raise Exception(error_message)

print(response)

# ---------------
# Retrieving data from existing data frames

data3 = """{
		"peripheral": [{
			"df": "PERIPHERAL"
		}],
		"population": {
			"column_01": [2.2, 3.2],
			"join_key": ["0", "0"],
			"time_stamp": [0.65, 0.81]
		}
	}"""

url = "http://localhost:1709/predict/MyModel/"

try:
    response = request.urlopen(
      url=url, 
      data=data.encode()
    ).read()
except error.HTTPError as e:
    error_message = e.read()
    raise Exception(error_message)

print(response)

# ---------------
# Selecting data from the data base.

peripheral_table.to_db("PERIPHERAL")

# A problem with SQL queries is that they often contain a
# quotation mark, which can create  conflicts with the JSON syntax. 
# Luckily, the json module in the Python standard library
# is smart enough to automatically escape quotation marks.
data4 = {
    "peripheral": [{
	"query": """SELECT * FROM "PERIPHERAL" WHERE "join_key" = '0';"""
    }],
    "population": {
	"column_01": [2.2, 3.2],
	"join_key": ["0", "0"],
	"time_stamp": [0.65, 0.81]
    }
}

data4_json = json.dumps(data4)
	
url = "http://localhost:1709/predict/MyModel/"

try:
    response = request.urlopen(
      url=url, 
      data=data.encode()
    ).read()
except error.HTTPError as e:
    error_message = e.read()
    raise Exception(error_message)

print(response)

# ----------------

engine.delete_project("examples")

