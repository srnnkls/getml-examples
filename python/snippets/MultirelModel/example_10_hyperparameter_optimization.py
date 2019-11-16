# Copyright 2019 The SQLNet Company GmbH

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

import numpy as np
import pandas as pd

import getml.aggregations as aggregations
import getml.datasets as datasets
import getml.engine as engine
import getml.hyperopt as hyperopt
import getml.loss_functions as loss_functions
import getml.models as models
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

population_table, peripheral_table = datasets.make_numerical(n_rows_population=1000)

# ----------------
# Upload data to the getML engine

peripheral_on_engine = engine.DataFrame(
    name="PERIPHERAL",
    join_keys=["join_key"],
    numerical=["column_01"],
    time_stamps=["time_stamp"]
)

peripheral_on_engine.send(
    peripheral_table
)

population_on_engine_training = engine.DataFrame(
    name="POPULATION_TRAINING",
    join_keys=["join_key"],
    numerical=["column_01"],
    time_stamps=["time_stamp"],
    targets=["targets"]
)

population_on_engine_training.send(
    population_table[:500]
)

population_on_engine_validation = engine.DataFrame(
    name="POPULATION_VALIDATION",
    join_keys=["join_key"],
    numerical=["column_01"],
    time_stamps=["time_stamp"],
    targets=["targets"]
)

population_on_engine_validation.send(
    population_table[500:]
)

# ----------------
# Build a reference model

population_placeholder = models.Placeholder(
    name="POPULATION"
)

peripheral_placeholder = models.Placeholder(
    name="PERIPHERAL"
)

population_placeholder.join(peripheral_placeholder, "join_key", "time_stamp")

predictor = predictors.LinearRegression()

model = models.MultirelModel(
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
# Build a hyperparameter space 

param_space = dict()

param_space['grid_factor'] = [1.0, 16.0]
param_space['min_num_samples'] = [100, 500]
param_space['num_features'] = [2, 10]
param_space['shrinkage'] = [0.0, 0.3]

# Any hyperparameters that relate to the predictor
# are preceded by "predictor__".
param_space['predictor__lambda'] = [0.0, 0.00001]

# ----------------
# Wrap a RandomSearch around the reference model

random_search = hyperopt.RandomSearch(
    model=model,
    param_space=param_space,
    n_iter=10
)

random_search.fit(
  population_table_training=population_on_engine_training,
  population_table_validation=population_on_engine_validation,
  peripheral_tables=[peripheral_on_engine]
)

# ----------------
# Wrap a LatinHypercubeSearch around the reference model

latin_search = hyperopt.LatinHypercubeSearch(
    model=model,
    param_space=param_space,
    n_iter=10
)

latin_search.fit(
  population_table_training=population_on_engine_training,
  population_table_validation=population_on_engine_validation,
  peripheral_tables=[peripheral_on_engine]
)
