++++
draft = true
type = "blogpost"
slug = "getstarted"
title = "Getting started"
description = ""
+++

This short blog post gets you started with **getML**. You will learn
the basic steps and commands to tackle your data science project using
its Python API. The underlying Python script can be accessed
[here](/there/).

# Motivation / Problem

To make our short introduction as realistic as possible, we will start
from a public-domain real-world data set. Let's pretend we are a
company selling and shipping a large portfolio of different
products. To both improve user experience and increase revenue, we
want to offer a special wrapping in case a product was bought as a
gift. But how to inform the costumers in question about our new
service without annoying everyone else?

The most prolific approach to solve such kind of problems is Machine
Learning. We will use it on the consumer expenditure public-use
microdata provided by the [U.S. Bureau of Labor
Statistics](https://www.bls.gov/cex/pumd_data.htm) to predict whether
a product is bought as a gift or not.


# Prerequisites

## Install getML

But first of all you have to install **getML**. Just go to our
[download page](/download-page/), choose the track that fits you most,
and unpack the tarball we provide. That's it.

## Run getML

To run the application, all you need to do is enter the unpacked
tarball and execute the `run` script.

```bash
./run
```

This starts up the **getML engine**, which was written in C++ for
efficiency and takes care of all the heavy lifting, and the **getML
monitor**, which serves as a convenient user interface. 

Next, you need to **log into the engine**. Open the up the web browser
of your choice and enter `localhost:1709` in the address bar. You will
access a *local* HTTP server run by the monitor, which will ask you to
enter your credentials or to create a new account. Please note that
this account is *not* a local one but the one you set up via our
homepage. Thus, you also need a working internet connection to run the
software.

## Install the getML Python3 API

Bundled with the **getML** binary you can also find its Python3 API. To
install it, use the following commands

```bash
cd python
python3 setup.py install
```

## Get the data

Finally, we need some data to work with. You have two
options here.

1. Get our cleaned and preprocessed [version](/link/data/) to start
right away (assumed in the remainder of the post).
2. Download the [original
dataset](https://www.bls.gov/cex/pumd_data.htm) (diary15) and perform
the preprocessing yourself using [this](/link/scripts/) cleaning
script.

# Staging the data

With the preprocessing already done we will start by setting a new
project in the **getML** engine and loading the prepared tables into
the Python environment.

```python
import pandas as pd
import getml.engine

engine.set_project("CE")

CE_population_training = pd.read_csv("../../../data/consumer_expenditure/CE_population_training.csv")
CE_population_validation = pd.read_csv("../../../data/consumer_expenditure/CE_population_validation.csv")
CE_peripheral = pd.read_csv("../../../data/consumer_expenditure/CE_peripheral.csv")

```

In order for the automated feature engineering to get the most out of
the data, we have to provided some additional information about its
content. If a column contains e.g. the type of a product encoded in
integers, operations like comparisons, summation, or the extraction
the maximum would most probably make no sense. It, therefore, needs to
be of recognized as *categorical* instead of *discrete*.

```python
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
```

We will also assign units to indicate which columns should be
compared and to fine-tune their handling. More information about this
subject can be found in the [long read](/link/long/) and the [API
documentation](/pai/doc/).

```python
units = dict()

units["UCC"] = "UCC"
units["UCC1"] = "UCC1"
units["UCC2"] = "UCC2"
units["UCC3"] = "UCC3"
units["UCC4"] = "UCC4"
units["UCC5"] = "UCC5"

units["EXPNYR"] = "year, comparison only"
```

With all that additional information in place we can finally construct
the `DataFrame`s, which will serve as our handles for the tables
stored in the engine. Using the `.send()` method we upload the
provided data to the engine and `.save()` ensures the `DataFrame` will
persist.

```python
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
```

# Building and fitting a model

## Placeholders

Now, all data is uploaded into the **getML** engine. But to train a model
using these tables, we still need a way to represent their relations
to each other.

We will do so with the concept of placeholders popularized by
Tensorflow and linking them using specific columns present in both
tables by calling the `.join()` method.

```python
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
```

For more information about this steps please have a look at [detailed description](/link/post/).

## Feature selector and predictor

Apart from our sophisticated algorithm for automated feature
engineering in relational data, **getML** has two other main
components. 

The first one is the feature selector, which picks the best set of
features from the generated ones. The second is the predictor, which
is trained on the features to make predictions and is the component
you already know from various other machine learning applications and
libraries.

For both instances we will use a XGBoost classifier.

```python
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

```

## Building a model

Finally, we have all pieces together to construct the overall
model. For details about its arguments, please have a look into the
[documentation](/getml/python/api/). Like a `DataFrame` a model needs
to be uploaded to the **getML engine** using the `.send()` method too.

```python
import getml.aggregations as aggregations
import getml.loss_functions as loss_functions

model = models.AutoSQLModel(
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

```

## Fitting the model

To build the features and train the predictor, all you need to do is
to call the `.fit()` method of the model.

```python
model = model.fit(
    population_table=df_population_training,
    peripheral_tables=[df_peripheral]
)
```

To see how well it performs, let's evaluate it on the validation set
using `.score()`.

```python
scores = model.score(
    population_table=df_population_validation,
    peripheral_tables=[df_peripheral]
)
```

Right now, **getML** supports six different scores: accuracy, AUC
(area under the ROC curve), and cross entropy for classification tasks
and MAE, RMSE, and R-squared (squared correlation coefficient) for
regression. Since determining whether a product was bought as a
present is a classification problem, we will recommend the AUC to
measure the performance of our model. If you wish, you can gather
additional data or tweak the parameters of the `AutoSQLModel` to
improve it even further.

As soon as you are satisfied with the performance of your model you
can use it in production to make predictions on new and unseen data
using `.predict()`.
