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

import getml.models.aggregations as aggregations
import getml.datasets as datasets
import getml.engine as engine
import getml.hyperopt as hyperopt
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

population_table, peripheral_table = datasets.make_numerical(n_rows_population=1000)
population_table_validation, _ = datasets.make_numerical(n_rows_population=1000)

population_placeholder = population_table.to_placeholder()
peripheral_placeholder = peripheral_table.to_placeholder()
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
# are preceded by "predictor_".
param_space['predictor_lambda'] = [0.0, 0.00001]

# ----------------
# Wrap a RandomSearch around the reference model

random_search = hyperopt.RandomSearch(
    model=model,
    param_space=param_space,
    n_iter=10
)

random_search.fit(
  population_table_training=population_table,
  population_table_validation=population_table_validation,
  peripheral_tables=[peripheral_table]
)

# ----------------
# Wrap a LatinHypercubeSearch around the reference model

latin_search = hyperopt.LatinHypercubeSearch(
    model=model,
    param_space=param_space,
    n_iter=10
)

latin_search.fit(
  population_table_training=population_table,
  population_table_validation=population_table_validation,
  peripheral_tables=[peripheral_table]
)

engine.delete_project("examples")
