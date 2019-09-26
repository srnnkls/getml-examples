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

## The overall Python3 script of the getting started guide.

## -------------------------------------------------------------------

import pandas as pd
import os

import getml.aggregations as aggregations
import getml.engine as engine
import getml.loss_functions as loss_functions
import getml.models as models
import getml.predictors as predictors


## -------------------------------------------------------------------

## Setting a project the getML engine will bundle all fitted models
## and saved DataFrames in.
engine.set_project("gettingStarted")

## -------------------------------------------------------------------

## Load all required data from disk.
source_path = os.path.join(os.getcwd(), "../../../data/consumer_expenditures/")

CE_population_training = pd.read_csv(os.path.join(source_path, "CE_population_training.csv"))
CE_population_validation = pd.read_csv(os.path.join(source_path, "CE_population_validation.csv"))
CE_peripheral = pd.read_csv(os.path.join(source_path, "CE_peripheral.csv"))

## -------------------------------------------------------------------
## Declare variables specifying the type of the individual columns to
## let the engine harness prior knowledge about the data at hand.

CATEGORICAL = [
    "UCC",
    "UCC1",
    "UCC2",
    "UCC3",
    "UCC4",
    "UCC5"]

DISCRETE = ["EXPNYR"]

JOIN_KEYS = [
    "NEWID",
    "BASKETID"]

NUMERICAL = ["COST"]

TARGETS = ["TARGET"]

TIME_STAMPS = [
    "TIME_STAMP",
    "TIME_STAMP_SHIFTED"]

## -------------------------------------------------------------------
## Set up units. This allows the engine to directly compare individual
## columns.

units = dict()

units["UCC"] = "UCC"
units["UCC1"] = "UCC1"
units["UCC2"] = "UCC2"
units["UCC3"] = "UCC3"
units["UCC4"] = "UCC4"
units["UCC5"] = "UCC5"

## Adding 'comparison only' to the unit forces the engine to only
## compare this column to others and forbids to use e.g. its absolute
## value in the creation of new features.
units["EXPNYR"] = "year, comparison only"

## -------------------------------------------------------------------
## Constructing getML DataFrames from the loaded data and upload them
## to the engine.

df_population_training = engine.DataFrame(
    "POPULATION_TRAINING",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS,
    units=units
).send(CE_population_training)
df_population_training.save()

df_population_validation = engine.DataFrame(
    "POPULATION_VALIDATION",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS,
    units=units
).send(CE_population_validation)
df_population_validation.save()

df_peripheral = engine.DataFrame(
    "PERIPHERAL",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS,
    units=units
).send(CE_peripheral)
df_peripheral.save()

## -------------------------------------------------------------------
## Build data model. Using the concept of placeholders, the model can
## store and use the relation between the data. For the consumer
## expenditure data set it consists of two self-joins.

CE_placeholder = models.Placeholder("PERIPHERAL")

CE_placeholder2 = models.Placeholder("PERIPHERAL")

CE_placeholder.join(
    CE_placeholder2,
    join_key="NEWID",
    time_stamp="TIME_STAMP",
    other_time_stamp="TIME_STAMP_SHIFTED")


CE_placeholder.join(
    CE_placeholder2,
    join_key="BASKETID",
    time_stamp="TIME_STAMP")

## -------------------------------------------------------------------
## Create two handles for XGBoost models used to a) select the best
## features from the generated ones and b) to be trained on the
## training set, validated on the validation set, and to perform the
## actual predictions.

feature_selector = predictors.XGBoostClassifier(
    booster="gbtree",
    n_estimators=100,
    n_jobs=6,
    max_depth=7,
    reg_lambda=500)

predictor = predictors.XGBoostClassifier(
    booster="gbtree",
    n_estimators=100,
    n_jobs=6,
    max_depth=7,
    reg_lambda=500)

## -------------------------------------------------------------------
## Construct an Multirel model and upload it to the getML engine.

model = models.MultirelModel(
    population=CE_placeholder,
    peripheral=[CE_placeholder],
    predictor=predictor,
    loss_function=loss_functions.CrossEntropyLoss(),
    aggregation=[
        aggregations.Avg,
        aggregations.Count,
        aggregations.CountDistinct,
        aggregations.CountMinusCountDistinct,
        aggregations.Max,
        aggregations.Median,
        aggregations.Min,
        aggregations.Sum
    ],
    use_timestamps=True,
    num_features=70,
    max_length=7,
    min_num_samples=100,
    shrinkage=0.1,
    grid_factor=1.0,
    regularization=0.0,
    round_robin=False,
    share_aggregations=0.04,
    share_conditions=0.8,
    sampling_factor=1.0
).send()

## -------------------------------------------------------------------
## Create features from the data using the Multirel algorithm and use
## them to train the predictor.

model = model.fit(
    population_table=df_population_training,
    peripheral_tables=[df_peripheral])

## -------------------------------------------------------------------
## Validate the trained model using the validation set.

scores = model.score(
    population_table = df_population_validation,
    peripheral_tables = [df_peripheral])
