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

import getml.aggregations as aggregations
import getml.engine as engine
import getml.loss_functions as loss_functions
import getml.models as models
import getml.predictors as predictors

#----------------

engine.set_project("examples")

#----------------
# Generate artificial dataset
#
# Don't worry - you don't really have to understand this part.
# This is just how we generate the example dataset. To learn more
# about Multirel just skip to "Build data model".

population_table = pd.DataFrame()
population_table["column_01"] = np.random.rand(500) * 2.0 - 1.0
population_table["join_key"] = range(500)
population_table["time_stamp_population"] = np.random.rand(500)

peripheral_table = pd.DataFrame()
peripheral_table["column_01"] = np.random.rand(5000) * 2.0 - 1.0
peripheral_table["join_key"] = [
    int(500.0 * np.random.rand(1)[0]) for i in range(5000)]
peripheral_table["join_key2"] = range(5000)
peripheral_table["time_stamp_peripheral"] = np.random.rand(5000)

peripheral_table2 = pd.DataFrame()
peripheral_table2["column_01"] = np.random.rand(125000) * 2.0 - 1.0
peripheral_table2["join_key2"] = [
    int(5000.0 * np.random.rand(1)[0]) for i in range(125000)]
peripheral_table2["time_stamp_peripheral2"] = np.random.rand(125000)

# ----------------
# Merge peripheral_table with peripheral_table2

temp = peripheral_table2.merge(
    peripheral_table[["join_key2", "time_stamp_peripheral"]],
    how="left",
    on="join_key2"
)

# Apply some conditions
temp = temp[
    (temp["time_stamp_peripheral2"] <= temp["time_stamp_peripheral"]) &
    (temp["time_stamp_peripheral2"] >= temp["time_stamp_peripheral"] - 0.5)
]

# Define the aggregation
temp = temp[["column_01", "join_key2"]].groupby(
    ["join_key2"],
    as_index=False
).count()

temp = temp.rename(index=str, columns={"column_01": "temporary"})

peripheral_table = peripheral_table.merge(
    temp,
    how="left",
    on="join_key2"
)

del temp

# Replace NaN with 0.0
peripheral_table["temporary"] = [
    0.0 if val != val else val for val in peripheral_table["temporary"]
]

# ----------------
# Merge population_table with peripheral_table

temp2 = peripheral_table.merge(
    population_table[["join_key", "time_stamp_population"]],
    how="left",
    on="join_key"
)

# Apply some conditions
temp2 = temp2[
    (temp2["time_stamp_peripheral"] <= temp2["time_stamp_population"])
]

# Define the aggregation
temp2 = temp2[["temporary", "join_key"]].groupby(
    ["join_key"],
    as_index=False
).mean()

temp2 = temp2.rename(index=str, columns={"temporary": "targets"})

population_table = population_table.merge(
    temp2,
    how="left",
    on="join_key"
)

del temp2

# Replace NaN targets with 0.0 - target values may never be NaN!.
population_table["targets"] = [
    0.0 if val != val else val for val in population_table["targets"]
]

# Remove temporary column.
del peripheral_table["temporary"]


# ----------------

population_table = population_table.rename(
    index=str, columns={"time_stamp_population": "time_stamp"})

peripheral_table = peripheral_table.rename(
    index=str, columns={"time_stamp_peripheral": "time_stamp"})

peripheral_table2 = peripheral_table2.rename(
    index=str, columns={"time_stamp_peripheral2": "time_stamp"})

# ----------------
# Build data model

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
    join_keys=["join_key", "join_key2"],
    time_stamps=["time_stamp"]
)

peripheral2_placeholder = models.Placeholder(
    name="PERIPHERAL2",
    numerical=["column_01"],
    join_keys=["join_key2"],
    time_stamps=["time_stamp"]
)

peripheral_placeholder.join(peripheral2_placeholder, "join_key2", "time_stamp")

population_placeholder.join(peripheral_placeholder, "join_key", "time_stamp")

predictor = predictors.LinearRegression()

model = models.MultirelModel(
    aggregation=[
        aggregations.Avg,
        aggregations.Count,
        aggregations.Sum
    ],
    population=population_placeholder,
    peripheral=[peripheral_placeholder, peripheral2_placeholder],
    loss_function=loss_functions.SquareLoss(),
    predictor=predictor,
    num_features=10,
    num_subfeatures=1,
    max_length=1,
    share_aggregations=1.0
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
