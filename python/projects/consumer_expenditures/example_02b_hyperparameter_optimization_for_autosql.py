import datetime
import os

import getml.aggregations as aggregations
import getml.engine as engine
import getml.hyperopt as hyperopt
import getml.loss_functions as loss_functions
import getml.predictors as predictors
import getml.models as models

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

# -----------------------------------------------------------------------------

engine.set_project("CE")

# -----------------------------------------------------------------------------
# Reload the data - if you haven't shut down the engine since loading the data
# in the first script, you can also call .refresh()

df_population_training = engine.DataFrame("POPULATION_TRAINING").load()

df_population_validation = engine.DataFrame("POPULATION_VALIDATION").load()

df_population_testing = engine.DataFrame("POPULATION_TESTING").load()

df_peripheral = engine.DataFrame("PERIPHERAL").load()

# -----------------------------------------------------------------------------
# Build data model - in this case, the data model is quite simple an consists
# of two self-joins

CE_placeholder = models.Placeholder("PERIPHERAL")

CE_placeholder2 = models.Placeholder("PERIPHERAL")

CE_placeholder.join(
    CE_placeholder2,
    join_key="NEWID",
    time_stamp="TIME_STAMP",
    other_time_stamp="TIME_STAMP_SHIFTED"
)

CE_placeholder.join(
    CE_placeholder2,
    join_key="BASKETID",
    time_stamp="TIME_STAMP"
)

# -----------------------------------------------------------------------------
# Set up the reference model - the data schema, the loss function and any
# hyperparameters that are not optimized will be taken from the reference
# model. 

feature_selector = predictors.XGBoostClassifier(
    booster="gbtree",
    n_estimators=100,
    n_jobs=6,
    max_depth=7,
    reg_lambda=500
)

predictor = predictors.XGBoostClassifier(
    booster="gbtree",
    n_estimators=100,
    n_jobs=6,
    max_depth=7,
    reg_lambda=500
)

model = models.AutoSQLModel(
    population=CE_placeholder,
    peripheral=[CE_placeholder],
    loss_function=loss_functions.CrossEntropyLoss(),
    aggregation=[
        aggregations.Avg,
        aggregations.Count,
        aggregations.CountDistinct,
        aggregations.CountMinusCountDistinct,
        aggregations.Max,
        aggregations.Median,
        aggregations.Min,
        aggregations.Sum,
        aggregations.Var
    ],
    feature_selector=feature_selector,
    predictor=predictor,
    allow_sets=True,
    num_threads=3
).send()

# ----------------
# Build a hyperparameter space 

param_space = dict()

param_space["grid_factor"]: [1.0, 16.0]
param_space["max_length"]: [1, 10]
param_space["min_num_samples"]: [100, 500]
param_space["num_features"]: [10, 500]
param_space["regularization"]: [0.0, 0.01]
param_space["share_aggregations"]: [0.01, 0.3]
param_space["share_selected_features"]: [0.1, 1.0]
param_space["shrinkage"]: [0.01, 0.4]

# Any hyperparameters that relate to the predictor
# are preceded by "predictor__".
param_space["predictor__n_estimators"] = [100, 400]
param_space["predictor__max_depth"] = [3, 15]
param_space["predictor__reg_lambda"] = [0.0, 1000.0]

# ----------------
# Wrap a latin hypercube search around the model.
# We have just added ten iterations - in practice it should
# be more.

latin_search = hyperopt.LatinHypercubeSearch(
    model=model,
    param_space=param_space,
    n_iter=2
)

latin_search.fit(
  population_table_training=df_population_training,
  population_table_validation=df_population_validation,
  peripheral_tables=[df_peripheral]
)

