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

time_series = pd.DataFrame()

time_series["join_key"] = np.zeros(1000).astype(int).astype(str)

time_series["time_stamp"] = np.arange(1000.0)
time_series["time_stamp_lagged"] = time_series["time_stamp"] - 1.0 

time_series["column_01"] = np.sin(np.pi*time_series["time_stamp"]/5.0) + time_series["time_stamp"]*0.1

# ----------------
# Upload data to the getML engine

population_on_engine = data.DataFrame(
    name="POPULATION",
    roles={
        "join_key": ["join_key"],
        "target": ["column_01"],
        "time_stamp": ["time_stamp_lagged"]}
).read_pandas(
    time_series
)

peripheral_on_engine = data.DataFrame(
    name="PERIPHERAL",
    roles={
        "join_key": ["join_key"],
        "numerical": ["column_01"],
        "time_stamp": ["time_stamp"]}
).read_pandas(
    time_series
)

# ----------------
# Build model

population_placeholder = data.Placeholder(
    name="TIME_SERIES"
)

peripheral_placeholder = data.Placeholder(
    name="TIME_SERIES"
)

population_placeholder.join(
  peripheral_placeholder, 
  join_key="join_key", 
  time_stamp="time_stamp_lagged",
  other_time_stamp="time_stamp",
)

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
    min_num_samples=1,
    num_features=10,
    share_aggregations=1.0,
    max_length=2,
    num_threads=4,
    delta_t=1.0 # Define the time delta
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
    population_table=population_on_engine,
    peripheral_tables=[peripheral_on_engine]
)

print(scores)

# ----------------

engine.delete_project("examples")
