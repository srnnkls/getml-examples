# Copyright 2018 The SQLNet Company GmbH

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

#----------------

engine.set_project("examples")

#----------------
# Generate artificial dataset

population_table, peripheral_table, peripheral_table2 = datasets.make_snowflake(
    aggregation1=aggregations.Avg,
    aggregation2=aggregations.Count
)

# ----------------
# Build data model

population_placeholder = population_table.to_placeholder()
peripheral_placeholder = peripheral_table.to_placeholder()
peripheral2_placeholder = peripheral_table2.to_placeholder()

peripheral_placeholder.join(peripheral2_placeholder, "join_key2", "time_stamp")
population_placeholder.join(peripheral_placeholder, "join_key", "time_stamp")

predictor = predictors.LinearRegression()

model = models.RelboostModel(
    population=population_placeholder,
    peripheral=[peripheral_placeholder, peripheral2_placeholder],
    loss_function=loss_functions.SquareLoss(),
    predictor=predictor,
    num_features=10,
    num_subfeatures=4,
    max_depth=1,
    gamma=0.0,
    reg_lambda=0.0,
    shrinkage=0.4,
    num_threads=0
).send()

# ----------------
# Fit model

model = model.fit(
    population_table=population_table,
    peripheral_tables=[peripheral_table, peripheral_table2]
)

features = model.transform(
    population_table=population_table,
    peripheral_tables=[peripheral_table, peripheral_table2]
)

yhat = model.predict(
    population_table=population_table,
    peripheral_tables=[peripheral_table, peripheral_table2]
)

print(model.to_sql())

scores = model.score(
    population_table=population_table,
    peripheral_tables=[peripheral_table, peripheral_table2]
)

print(scores)

# ----------------

engine.delete_project("examples")


