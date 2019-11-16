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

population_table, peripheral_table = datasets.make_numerical()

# ----------------
# Build model

population_placeholder = models.Placeholder(
    name="POPULATION",
    numerical=["column_01"],
    join_keys=["join_key"],
    time_stamps=["time_stamp"],
    targets=["targets"]
)

peripheral_placeholder = models.Placeholder(
    name="PERIPHERAL",
    numerical=["column_01"],
    join_keys=["join_key"],
    time_stamps=["time_stamp"]
)

population_placeholder.join(peripheral_placeholder, "join_key", "time_stamp")

predictor = predictors.LinearRegression()

model = models.RelboostModel(
    population=population_placeholder,
    peripheral=[peripheral_placeholder],
    loss_function=loss_functions.SquareLoss(),
    predictor=predictor,
    num_features=10,
    max_depth=1,
    reg_lambda=0.0,
    shrinkage=0.3,
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

scores = model.score(
    population_table=population_table,
    peripheral_tables=[peripheral_table]
)

print(scores)

# ----------------