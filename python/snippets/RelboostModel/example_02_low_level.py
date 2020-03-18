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
import getml.data as data
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


n_rows_population = 500
n_rows_peripheral = 125000
aggregation = aggregations.Count

random = np.random.RandomState(8290)

population_table = pd.DataFrame()
population_table["column_01"] = random.rand(n_rows_population) * 2.0 - 1.0
population_table["join_key"] = np.arange(n_rows_population)
population_table["time_stamp_population"] = random.rand(n_rows_population)

peripheral_table = pd.DataFrame()
peripheral_table["column_01"] = random.rand(n_rows_peripheral) * 2.0 - 1.0
peripheral_table["join_key"] = random.randint(0, n_rows_population, n_rows_peripheral) 
peripheral_table["time_stamp_peripheral"] = random.rand(n_rows_peripheral)

# Compute targets
temp = peripheral_table.merge(
    population_table[["join_key", "time_stamp_population"]],
    how="left",
    on="join_key"
)

# Apply some conditions
temp = temp[
    (temp["time_stamp_peripheral"] <= temp["time_stamp_population"]) &
    (temp["time_stamp_peripheral"] >= temp["time_stamp_population"] - 0.5)
]

# Define the aggregation
temp = datasets._aggregate(temp, aggregation, "column_01", "join_key")

temp = temp.rename(index=str, columns={"column_01": "targets"})

population_table = population_table.merge(
    temp,
    how="left",
    on="join_key"
)

del temp

population_table = population_table.rename(
    index=str, columns={"time_stamp_population": "time_stamp"})

peripheral_table = peripheral_table.rename(
    index=str, columns={"time_stamp_peripheral": "time_stamp"})

# Replace NaN targets with 0.0 - target values may never be NaN!.
population_table.targets = np.where(
        np.isnan(population_table['targets']), 
        0, 
        population_table['targets'])

# ----------------
# Upload data to the getML engine

# The low-level API allows you to upload
# data to the getML engine in a piecewise fashion.
# Here we load the first part of the pandas.DataFrame...
peripheral_on_engine = data.DataFrame(
    name="PERIPHERAL",
    roles={
        "join_key": ["join_key"],
        "numerical": ["column_01"],
        "time_stamp": ["time_stamp"]}
).read_pandas(
    peripheral_table[:2000]
)

# ...and now we load the second part
peripheral_on_engine.read_pandas(
    peripheral_table[2000:],
    append=True
)

# The low-level API allows you to upload
# data to the getML engine in a piecewise fashion.
# Here we load the first part of the pandas.DataFrame...
population_on_engine = data.DataFrame(
    name="POPULATION",
    roles={
        "join_key": ["join_key"],
        "numerical": ["column_01"],
        "time_stamp": ["time_stamp"],
        "target": ["targets"]}
).read_pandas(
    population_table[:20]
)

# ...and now we load the second part
population_on_engine.read_pandas(
   population_table[20:],
   append=True
)

# ----------------
# Build model

population_placeholder = data.Placeholder(
    name="POPULATION"
)

peripheral_placeholder = data.Placeholder(
    name="PERIPHERAL"
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
    population_table=population_on_engine,
    peripheral_tables=[peripheral_on_engine]
)

# ----------------

features = model.transform(
    population_table=population_on_engine,
    peripheral_tables=[peripheral_on_engine]
)

# ----------------

yhat = model.predict(
    population_table=population_on_engine,
    peripheral_tables=[peripheral_on_engine]
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

engine.delete_project("examples")
