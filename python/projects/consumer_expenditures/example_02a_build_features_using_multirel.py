import datetime
import os

import getml.models.aggregations as aggregations
import getml.data as data 
import getml.engine as engine
import getml.models.loss_functions as loss_functions
import getml.data.placeholder as placeholder
import getml.predictors as predictors
import getml.models as models

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

#predictor = predictors.LogisticRegression()

model = models.MultirelModel(
    population=population_placeholder,
    peripheral=[expd_placeholder, memd_placeholder],
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
    num_features=10,
    share_aggregations=0.2,
    #feature_selector=feature_selector,
    predictor=predictor,
    allow_sets=True,
    max_length=3,
    min_num_samples=200,
    num_threads=4,
    shrinkage=0.1,
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
