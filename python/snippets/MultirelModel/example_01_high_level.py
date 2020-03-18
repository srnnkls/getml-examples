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
# SELECT COUNT( * )
# FROM POPULATION t1
# LEFT JOIN PERIPHERAL t2
# ON t1.join_key = t2.join_key
# WHERE (
#    ( t1.time_stamp - t2.time_stamp <= 0.5 )
# ) AND t2.time_stamp <= t1.time_stamp
# GROUP BY t1.join_key,
#          t1.time_stamp;
#

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
# Build model

population_placeholder = data.Placeholder(
    name="POPULATION",
    numerical=["column_01"],
    join_keys=["join_key"],
    time_stamps=["time_stamp"],
    targets=["targets"]
)

peripheral_placeholder = data.Placeholder(
    name="PERIPHERAL",
    numerical=["column_01"],
    join_keys=["join_key"],
    time_stamps=["time_stamp"]
)

population_placeholder.join(peripheral_placeholder, "join_key", "time_stamp")

predictor = predictors.LinearRegression()

#predictor = predictors.XGBoostRegressor()

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
    max_length=3,
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

features = model.transform(
    population_table=population_table,
    peripheral_tables=[peripheral_table],
    df_name="features"
)

print(features)

# ----------------

yhat = model.predict(
    population_table=population_table,
    peripheral_tables=[peripheral_table]
)

# ----------------

#print(model.to_sql())

# ----------------

scores = model.score(
    population_table=population_table,
    peripheral_tables=[peripheral_table]
)

print(scores)

# ----------------

engine.delete_project("examples")
