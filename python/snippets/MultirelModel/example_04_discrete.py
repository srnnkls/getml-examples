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
# SELECT MIN( t2.column_01 )
# FROM POPULATION t1
# LEFT JOIN PERIPHERAL t2
# ON t1.join_key = t2.join_key
# WHERE (
#    ( t2.column_01 > 0 )
# ) AND t2.time_stamp <= t1.time_stamp
# GROUP BY t1.join_key,
#          t1.time_stamp;


population_table, peripheral_table = datasets.make_discrete()

population_placeholder = population_table.to_placeholder()
peripheral_placeholder = peripheral_table.to_placeholder()
population_placeholder.join(peripheral_placeholder, "join_key", "time_stamp")

predictor = predictors.LinearRegression()

model = models.MultirelModel(
    aggregation=[
        aggregations.Avg,
        aggregations.Count,
        aggregations.Max,
        aggregations.Min,
        aggregations.Sum
    ],
    population=population_placeholder,
    peripheral=[peripheral_placeholder],
    loss_function=loss_functions.SquareLoss(),
    predictor=predictor,
    num_features=10,
    share_aggregations=1.0,
    max_length=3,
    num_threads=4
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

scores = model.score(
    population_table=population_table,
    peripheral_tables=[peripheral_table]
)

print(scores)

# ----------------

engine.delete_project("examples")
