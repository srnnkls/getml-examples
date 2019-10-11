#!/usr/bin/env python
# coding: utf-8

# # Get Started with getML

# This guide gets you started with [getML](https://get.ml). You will learn
# the basic steps and commands to tackle your data science project using
# the [getML Python API](https://github.com/getml/getml-python-api). A detailed explanation of the dataset and the methods used below can be found in the [complete tutorial](https://docs.get.ml/latest/tutorial/consumer_expenditure/index.html).

# ## Introduction
# 
# Let's pretend we are a company selling and shipping a large portfolio of different
# products. To improve the experience of our customers and increase our revenue, we
# want to offer a special wrapping in case a product was bought as a
# gift. At the same time we only want to offer this service to customers who buy their products as gifts and not annoy the majority of our customers who do not interested in this special service. In order to do that we need to predict when a customer buys a product as a gift.
# 
# We will use it on the consumer expenditure public-use
# microdata provided by the [U.S. Bureau of Labor
# Statistics](https://www.bls.gov/cex/pumd_data.htm) to predict whether
# a product is bought as a gift or not. We have preprocessed the data for this introductory example using [this script](../../data/consumer_expenditures/raw/convert_CE_data.py).
# 
# 
# ### Prerequisites
# 
# There's a few things you need to do before you can dive into the actual task.
# 
# * **Install getML**
# 
# Just go to our [webpage](https://get.ml), browse to the download page, choose the track
# that fits you most, and unpack the tarball we provide. That's it.
# 
# * **Start the getML engine**
# 
# All you need to do is enter the unpacked
# tarball and execute the `run` script.
# 
# ```bash
# ./run
# ```
# 
# This start the getML engine, the C++ backend of getML that's responsible for all the heavy lifting, and the getML monitor a convenient interface to the engine.
# 
# * **Log in**
# 
# Open the up a web browser
# and enter `localhost:1709` in the address bar. You will
# access a *local* HTTP server run by the getML monitor, which will ask you to
# enter your credentials or to create a new account.
# 
# 
# * **Install the Python API**
# 
# You can install the getML Python API from PyPI
# 
# ```bash
# pip install getml
# ```

# 
# ## Staging the data
# 
# We will start by starting a new project in the getML engine and loading
# the prepared data tables into the Python environment.

# In[ ]:


import os
import pandas as pd
import getml.engine as engine

engine.set_project("gettingStarted")

# Location inside this repository the data is kept.
source_path = os.path.join(os.getcwd(), "../../../data/consumer_expenditures/")

CE_population_training = pd.read_csv(os.path.join(source_path, "CE_population_training.csv"))
CE_population_validation = pd.read_csv(os.path.join(source_path, "CE_population_validation.csv"))
CE_peripheral = pd.read_csv(os.path.join(source_path, "CE_peripheral.csv"))


# In order for the automated feature engineering to get the most out of
# the data, we have to provided some additional information about its
# content. If a column contains e.g. the type of a product encoded in
# integers, operations like comparisons, summation, or the extraction
# the maximum would most probably make no sense. It, therefore, needs to
# be of recognized as *categorical* instead of *discrete*.

# In[ ]:


# Product categories
CATEGORICAL = [
    "UCC",
    "UCC1",
    "UCC2",
    "UCC3",
    "UCC4",
    "UCC5"]

# Year of purchase
DISCRETE = ["EXPNYR"]

# Join keys
JOIN_KEYS = [
    "NEWID",
    "BASKETID"]

# Price
NUMERICAL = ["COST"]

# Gift/no gift
TARGETS = ["TARGET"]

# Time stamps
TIME_STAMPS = [
    "TIME_STAMP",
    "TIME_STAMP_SHIFTED"]


# We will also assign units to indicate which columns should be
# compared and to fine-tune their handling. For more information on this please go to the complete [tutorial](https://docs.get.ml/latest/tutorial/consumer_expenditure/index.html).

# In[ ]:


units = dict()

units["UCC"] = "UCC"
units["UCC1"] = "UCC1"
units["UCC2"] = "UCC2"
units["UCC3"] = "UCC3"
units["UCC4"] = "UCC4"
units["UCC5"] = "UCC5"

units["EXPNYR"] = "year, comparison only"


# With this additional information in place we can construct
# the `DataFrame`s, which will serve as our handles for the tables
# stored in the engine. Using the `.send()` method we upload the
# provided data to the engine and `.save()` ensures the `DataFrame` will
# persist.

# In[ ]:


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


# ## Building and fitting a model
# 
# ### Placeholders
# 
# Now, all data is uploaded into the getML engine. To train a model
# using these tables, we now need a way to represent their relations
# to each other.
# 
# We will do so with the concept of placeholders popularized by
# Tensorflow and linking them using specific columns present in both
# tables by calling the `.join()` method.

# In[ ]:


import getml.models as models

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


# For more information about this steps please have a look at the detailed description in the [tutorial](https://docs.get.ml/latest/tutorial/consumer_expenditure/ce_train_single_multirel_model.html).
# 
# ### Feature selector and predictor
# 
# Apart from our sophisticated algorithm for automated feature
# engineering in relational data, getML has two other main
# components. 
# 
# The first one is the feature selector, which picks the best set of
# features from the generated ones. The second is the predictor, which
# is trained on the features to make predictions. This is the component
# you already know from various other machine learning applications and
# libraries.
# 
# For both instances we will use a XGBoost classifier.

# In[ ]:


import getml.predictors as predictors

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


# ### Building a model
# 
# Finally, we have all pieces together to construct the overall
# model. For details about its arguments, please have a look into the
# [documentation](https://docs.get.ml). Like a `DataFrame` a model needs
# to be uploaded to the getML engine using the `.send()` method too.

# In[ ]:


import getml.aggregations as aggregations
import getml.loss_functions as loss_functions

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


# ### Fitting the model
# 
# To build the features and train the predictor, all you need to do is
# to call the `.fit()` method of the model.

# In[ ]:


model = model.fit(
    population_table=df_population_training,
    peripheral_tables=[df_peripheral]
)


# To see how well it performs, let's evaluate it on the validation set
# using `.score()`.

# In[ ]:


scores = model.score(
    population_table=df_population_validation,
    peripheral_tables=[df_peripheral]
)

print(scores)


# For the time beeing, getML supports six different scores: accuracy, AUC
# (area under the ROC curve), and cross entropy for classification tasks
# and MAE, RMSE, and R-squared (squared correlation coefficient) for
# regression. Since determining whether a product was bought as a
# present is a classification problem, we will recommend the AUC to
# measure the performance of our model. If you wish, you can gather
# additional data or tweak the parameters of the `MultirelModel` to
# improve it even further.
# 
# As soon as you are satisfied with the performance of your model you
# can use it in production to make predictions on new and unseen data
# using `.predict()`.

# ### Next steps
# 
# This guide has shown you the very basics of getML. If you're interested in the software in general, head over to the [getML webpage](https://get.ml). If you're curious about other features of getML go to the [technical documentation](https://docs.get.ml). If you want to know more about the consumer expenditure analysis presented above, go through the extensive [tutorial](https://docs.get.ml/latest/tutorial/consumer_expenditure/index.html).
