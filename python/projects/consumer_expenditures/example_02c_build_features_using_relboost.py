import datetime
import os

import getml.data as data 
import getml.engine as engine
import getml.models.loss_functions as loss_functions
import getml.data.placeholder as placeholder
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

df_population_training = data.load_data_frame("POPULATION_TRAINING")

df_population_validation = data.load_data_frame("POPULATION_VALIDATION")

df_population_testing = data.load_data_frame("POPULATION_TESTING")

df_expd = data.load_data_frame("EXPD")

df_memd = data.load_data_frame("MEMD")

# -----------------------------------------------------------------------------
# Build data model

population_placeholder = placeholder.Placeholder("POPULATION")

expd_placeholder = placeholder.Placeholder("EXPD")

memd_placeholder = placeholder.Placeholder("MEMD")

population_placeholder.join(
    expd_placeholder,
    join_key="NEWID",
    time_stamp="TIME_STAMP"
)

population_placeholder.join(
    memd_placeholder,
    join_key="NEWID",
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
    reg_lambda=0.0
)

#predictor = predictors.GradientBoostingClassifier(
#    max_depth=7,
#    reg_lambda=0.0
#)

#predictor = predictors.LogisticRegression()

model = models.RelboostModel(
    population=population_placeholder,
    peripheral=[expd_placeholder, memd_placeholder],
    loss_function=loss_functions.CrossEntropyLoss(),
    shrinkage=0.1,
    gamma=0.0,
    min_num_samples=200,
    num_features=20,
    share_selected_features=0.0,
    reg_lambda=0.0,
    sampling_factor=1.0,
    predictor=predictor,
    #feature_selector=feature_selector,
    num_threads=0,
    include_categorical=True
).send()

# -----------------------------------------------------------------------------
# Fit model

model = model.fit(
    population_table=df_population_training,
    peripheral_tables=[df_expd, df_memd]
)

# -----------------------------------------------------------------------------
# Show SQL code

print(model.to_sql())

# -----------------------------------------------------------------------------
# Score model

scores = model.score(
    population_table=df_population_training,
    peripheral_tables=[df_expd, df_memd]
)

print("In-sample:")
print(scores)
print()

scores = model.score(
    population_table=df_population_validation,
    peripheral_tables=[df_expd, df_memd]
)

print("Out-of-sample:")
print(scores)
print()

# -----------------------------------------------------------------------------
# Get targets, for comparison

target = df_population_validation.to_pandas()["TARGET"]

# -----------------------------------------------------------------------------
# Get the features

features = model.transform(
    population_table=df_population_validation,
    peripheral_tables=[df_expd, df_memd]
)

# -----------------------------------------------------------------------------
# Get predictions

predictions = model.predict(
    population_table=df_population_validation,
    peripheral_tables=[df_expd, df_memd]
)
