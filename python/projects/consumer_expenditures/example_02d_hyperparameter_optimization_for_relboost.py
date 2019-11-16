import datetime
import os

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
# Set hyperparameters - this is just for demonstration purposes. You are very
# welcome to play with the hyperparameters to get better results. For instance,
# increasing num_features should get you over an AUC of 0.8.

feature_selector = predictors.XGBoostClassifier(
    booster="gbtree",
    n_estimators=100,
    n_jobs=6,
    max_depth=7,
    reg_lambda=500
)

#feature_selector = predictors.LogisticRegression()

predictor = predictors.XGBoostClassifier(
    booster="gbtree",
    n_estimators=100,
    n_jobs=6,
    max_depth=7,
    reg_lambda=500
)

#predictor = predictors.LogisticRegression()

model = models.RelboostModel(
    population=CE_placeholder,
    peripheral=[CE_placeholder],
    loss_function=loss_functions.CrossEntropyLoss(),
    shrinkage=0.1,
    gamma=0.0,
    min_num_samples=200,
    num_features=20,
    share_selected_features=1.0,
    reg_lambda=0.01,
    sampling_factor=1.0,
    predictor=predictor,
    feature_selector=feature_selector,
    num_threads=6
).send()

# ----------------
# Build a hyperparameter space 

param_space = dict()

param_space['max_depth'] = [3, 10]
param_space['min_num_samples'] = [100, 500]
param_space['num_features'] = [20, 200]
param_space['reg_lambda'] = [0.0001, 0.1]
param_space['share_selected_features'] = [0.1, 1.0]
param_space['shrinkage'] = [0.01, 0.3]

# ----------------
# Wrap a latin hypercube search around the model

latin_search = hyperopt.LatinHypercubeSearch(
    model=model,
    param_space=param_space,
    n_iter=10
)

latin_search.fit(
  population_table_training=df_population_training,
  population_table_validation=df_population_validation,
  peripheral_tables=[df_peripheral]
)


