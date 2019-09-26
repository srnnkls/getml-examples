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
# SELECT AVG( t2.column_02 )
# FROM POPULATION t1
# LEFT JOIN PERIPHERAL t2
# ON t1.join_key = t2.join_key
# WHERE (
#    ( t1.column_01 = t2.column_01 )
# ) AND t2.time_stamp <= t1.time_stamp
# GROUP BY t1.join_key,
#          t1.time_stamp;
#
# Don't worry - you don't really have to understand this part.
# This is just how we generate the example dataset. To learn more
# about Multirel just skip to "Build model".

population_table = pd.DataFrame()
population_table["column_01_population"] = (np.random.rand(
    500)*10.0).astype(np.int).astype(np.str)
population_table["join_key"] = range(500)
population_table["time_stamp_population"] = np.random.rand(500)

peripheral_table = pd.DataFrame()
peripheral_table["column_01_peripheral"] = (np.random.rand(
    125000)*10.0).astype(np.int).astype(np.str)
peripheral_table["column_02"] = np.random.rand(125000) * 2.0 - 1.0
peripheral_table["join_key"] = [
    int(500.0 * np.random.rand(1)[0]) for i in range(125000)]
peripheral_table["time_stamp_peripheral"] = np.random.rand(125000)

# ----------------

temp = peripheral_table.merge(
    population_table[["join_key", "time_stamp_population",
    "column_01_population"]],
    how="left",
    on="join_key"
)

# Apply some conditions
temp = temp[
    (temp["time_stamp_peripheral"] <= temp["time_stamp_population"]) &
    (temp["column_01_peripheral"] == temp["column_01_population"])
]

# Define the aggregation
temp = temp[["column_02", "join_key"]].groupby(
    ["join_key"],
    as_index=False
).mean()

temp = temp.rename(index=str, columns={"column_02": "targets"})

population_table = population_table.merge(
    temp,
    how="left",
    on="join_key"
)

population_table = population_table.rename(
  index=str, 
  columns={"column_01_population": "column_01"}
)

peripheral_table = peripheral_table.rename(
  index=str, 
  columns={"column_01_peripheral": "column_01"}
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

# ----------------
# Upload data to the getML engine

units = dict()

units["column_01"] = "unit_01"

peripheral_on_engine = engine.DataFrame(
    name="PERIPHERAL",
    join_keys=["join_key"],
    categorical=["column_01"],
    numerical=["column_02"],
    time_stamps=["time_stamp"],
    units=units # You can set the units here...
)

peripheral_on_engine.send(
    peripheral_table
)

population_on_engine = engine.DataFrame(
    name="POPULATION",
    join_keys=["join_key"],
    categorical=["column_01"],
    time_stamps=["time_stamp"],
    targets=["targets"],
    units=units # You can set the units here...
)

population_on_engine.send(
    population_table
)

# ... or you can declare them after loading the data.
population_on_engine.categorical("column_01").set_unit("unit_01")
peripheral_on_engine.categorical("column_01").set_unit("unit_01")

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

model = models.MultirelModel(
    aggregation=[
        aggregations.Avg,
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

scores = model.score(
    population_table=population_table,
    peripheral_tables=[peripheral_table]
)

print(scores)

# ----------------
