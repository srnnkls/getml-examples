# getml-examples

This repository contains various examples and tutorials to help you to get started with
**relational learning** in general and **getML** in particular.

Relational learning is about **machine learning specifically designed for
relational data**. Since most data scientists work with business data and 
practically all business data is relational (time series are a special
case of relational data), **relational learning is an important skill
for any data scientist to have**.

For more information on getML, check out the [official documentation](https://docs.getml.com).

To **download getML for free**, click [here](https://getml.com/product).

## Projects

This repository contains **completely self-contained** iPython notebooks.

You are actively encouraged to try them out, **reproduce our results** and 
**use them as a blueprint for your own projects**:

* [Getting started](https://github.com/getml/getml-examples/blob/master/python/projects/getting_started/getting_started.ipynb)
  - this notebook runs you through the most basic steps of relational learning using getML. 
    It is based on an artificial dataset. 

* [Loans](https://github.com/getml/getml-examples/blob/master/python/projects/loans/loans.ipynb)
  - the loans data set is one of the most commonly used data sets in the relational learning literature.
    Using getML's MultirelModel, we show how you can outperform practically all peer-reviewed academic papers 
    based on the loans data set.

* [Occupancy detection](https://github.com/getml/getml-examples/blob/master/python/projects/occupancy_detection/occupancy_detection.ipynb)
  - the occupancy detection data set is a **very simple multivariate time series**. This is to demonstrate how
    relational learning can be successfully applied to time series.

* [Consumer expenditures](https://github.com/getml/getml-examples/blob/master/python/projects/consumer_expenditures/consumer_expenditures.ipynb)
  - the consumer expenditures data set is about analyzing consumer's consumption patterns to predict whether an item was purchased
    as a gift. Using getML's relational boosting algorithm, we can reach an out-of-sample AUC of over 90%.

