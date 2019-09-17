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
# FROM POPULATION_TABLE t1
# LEFT JOIN PERIPHERAL_TABLE t2
# ON t1.join_key = t2.join_key
# WHERE (
#    ( t2.column_01 != '1' AND t2.column_01 != '2' AND t2.column_01 != '9' )
# ) AND t2.time_stamps <= t1.time_stamps
# GROUP BY t2.join_key;
#
# Don't worry - you don't really have to understand this part.
# This is just how we generate the example dataset. To learn more
# about getML just skip to "Build model".

population_table = pd.DataFrame()
population_table["column_01"] = (np.random.rand(
    500)*10.0).astype(np.int).astype(np.str)
population_table["join_key"] = range(500)
population_table["time_stamp_population"] = np.random.rand(500)

peripheral_table = pd.DataFrame()
peripheral_table["column_01"] = (np.random.rand(
    125000)*10.0).astype(np.int).astype(np.str)
peripheral_table["join_key"] = [
    int(500.0 * np.random.rand(1)[0]) for i in range(125000)]
peripheral_table["time_stamp_peripheral"] = np.random.rand(125000)

# ----------------

temp = peripheral_table.merge(
    population_table[["join_key", "time_stamp_population"]],
    how="left",
    on="join_key"
)

# Apply some conditions
temp = temp[
    (temp["time_stamp_peripheral"] <= temp["time_stamp_population"]) &
    (temp["column_01"] != "1") &
    (temp["column_01"] != "2") &
    (temp["column_01"] != "9")
]

# Define the aggregation
temp = temp[["column_01", "join_key"]].groupby(
    ["join_key"],
    as_index=False
).count()

temp = temp.rename(index=str, columns={"column_01": "targets"})

population_table = population_table.merge(
    temp,
    how="left",
    on="join_key"
)

del temp

# ----------------

population_table = population_table.rename(
    index=str, columns={"time_stamp_population": "time_stamp"})

peripheral_table = peripheral_table.rename(
    index=str, columns={"time_stamp_peripheral": "time_stamp"})

# ----------------

# Replace NaN targets with 0.0 - target values may never be NaN!.
population_table["targets"] = [
    0.0 if val != val else val for val in population_table["targets"]
]

#----------------
# Upload data to the getML engine

peripheral_on_engine = engine.DataFrame(
    name="PERIPHERAL",
    join_keys=["join_key"],
    categorical=["column_01"],
    time_stamps=["time_stamp"]
)

# The low-level API allows you to upload
# data to the AutoSQL engine in a piecewise fashion.
# Here we load the first part of the pandas.DataFrame...
peripheral_on_engine.send(
    peripheral_table[:2000]
)

# ...and now we load the second part
peripheral_on_engine.append(
    peripheral_table[2000:]
)

population_on_engine = engine.DataFrame(
    name="POPULATION",
    join_keys=["join_key"],
    time_stamps=["time_stamp"],
    targets=["targets"]
)

# The low-level API allows you to upload
# data to the AutoSQL engine in a piecewise fashion.
# Here we load the first part of the pandas.DataFrame...
population_on_engine.send(
    population_table[:20]
)

# ...and now we load the second part
population_on_engine.append(
   population_table[20:]
)

# ----------------
# Build model

population_placeholder = models.Placeholder(
    name="POPULATION"
)

peripheral_placeholder = models.Placeholder(
    name="PERIPHERAL"
)

population_placeholder.join(peripheral_placeholder, "join_key", "time_stamp")

predictor = predictors.LinearRegression()

model = models.AutoSQLModel(
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
