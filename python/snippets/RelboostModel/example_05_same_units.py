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
import getml.datasets as datasets
import getml.engine as engine
import getml.models.loss_functions as loss_functions
import getml.models as models
import getml.data as data
import getml.predictors as predictors

# --------------------------------------------------------------------

def _fit_model(population_df, peripheral_df, population_ph, peripheral_ph, seed, units):

    # ----------------------------------------------------------------
    
    predictor = predictors.LinearRegression()
    
    # ----------------------------------------------------------------
    
    model = models.RelboostModel(
        population=population_ph,
        peripheral=[peripheral_ph],
        loss_function=loss_functions.SquareLoss(),
        predictor=predictor,
        num_features=10,
        max_depth=1,
        reg_lambda=0.0,
        shrinkage=0.3,
        num_threads=1,
        seed=seed,
        units=units
    ).send()

    # ----------------------------------------------------------------

    model = model.fit(
        population_table=population_df,
        peripheral_tables=[peripheral_df]
    )
    
    # ----------------------------------------------------------------

    features = model.transform(
        population_table=population_df,
        peripheral_tables=[peripheral_df]
    )

    # ----------------------------------------------------------------

    yhat = model.predict(
        population_table=population_df,
        peripheral_tables=[peripheral_df]
    )

    # ----------------------------------------------------------------

    scores = model.score(
        population_table=population_df,
        peripheral_tables=[peripheral_df]
    )

    # ----------------------------------------------------------------
    
    return model, features, yhat, scores

# --------------------------------------------------------------------

def test_relboost_same_units():
    """Check if the same results will be obtained regardless of whether
    the units are assigned to the DataFrame, to the Columns, or to the
    RelboostModel.

    """
    
    # ----------------------------------------------------------------
    
    engine.set_project("examples")
    
    seed = 33231
    
    units = {"column_01": "column_01"}

    # ----------------------------------------------------------------

    # Generate artificial dataset
    # The problem we create looks like this:
    #
    # SELECT COUNT( * )
    # FROM POPULATION t1
    # LEFT JOIN PERIPHERAL t2
    # ON t1.join_key = t2.join_key
    # WHERE (
    #    ( t1.time_stamp - t2.time_stamp <= 0.5 )
    # ) AND t2.time_stamp <= t1.time_stamp
    # GROUP BY t1.join_key,
    #          t1.time_stamp;

    # ----------------------------------------------------------------
    
    # Assign the units to the columns
    population_table_columns, peripheral_table_columns = datasets.make_same_units_numerical(random_state = seed)

    population_table_columns.set_unit("column_01", "column_01")
    peripheral_table_columns.set_unit("column_01", "column_01")

    population_placeholder_columns = population_table_columns.to_placeholder()
    peripheral_placeholder_columns = peripheral_table_columns.to_placeholder()
    population_placeholder_columns.join(peripheral_placeholder_columns, "join_key", "time_stamp")

    # ----------------------------------------------------------------
    
    model_columns, features_columns, yhat_columns, scores_columns = _fit_model(
        population_table_columns, peripheral_table_columns,
        population_placeholder_columns, peripheral_placeholder_columns,
        seed, dict())
     
    # ----------------------------------------------------------------

    # Assign units to Model
    population_table_model, peripheral_table_model = datasets.make_same_units_numerical(random_state = seed)

    population_placeholder_model = population_table_model.to_placeholder()
    peripheral_placeholder_model = peripheral_table_model.to_placeholder()
    population_placeholder_model.join(peripheral_placeholder_model, "join_key", "time_stamp")

    # ----------------------------------------------------------------

    model_model, features_model, yhat_model, scores_model = _fit_model(
        population_table_model, peripheral_table_model,
        population_placeholder_model, peripheral_placeholder_model,
        seed, units)
    
    # ----------------------------------------------------------------

    # Check whether the results are the same.
    assert scores_model == scores_columns
    
    assert (yhat_model == yhat_columns).all()
     
    # ----------------------------------------------------------------
    
    engine.delete_project("examples")
    
    # ----------------------------------------------------------------
