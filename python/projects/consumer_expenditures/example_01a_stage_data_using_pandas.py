import datetime
import os
import time

import numpy as np
import pandas as pd

import getml.engine as engine

# -----------------------------------------------------------------------------

engine.set_project("CE")

# -----------------------------------------------------------------------------
# Begin timing

begin = time.time()

# -----------------------------------------------------------------------------
# Read the data from the source files
source_path = os.path.join(os.getcwd(), "../../../data/consumer_expenditures/raw/")

expd = pd.read_csv(os.path.join(source_path, "expd151.csv"))
expd = expd.append(pd.read_csv(os.path.join(source_path, "expd152.csv")))
expd = expd.append(pd.read_csv(os.path.join(source_path, "expd153.csv")))
expd = expd.append(pd.read_csv(os.path.join(source_path, "expd154.csv")))

# -----------------------------------------------------------------------------
# Set up target - we want to predict whether the item is a gift

expd["TARGET"] = [0.0 if elem == 2 else 1.0 for elem in expd["GIFT"]]

# -----------------------------------------------------------------------------
# Remove the instances where date is nan - they will be ignored by the Multirel
# engine anyway, because of the NULL value handling policy.

expd = expd[
    (expd["EXPNYR"] == expd["EXPNYR"]) & (expd["EXPNMO"] == expd["EXPNMO"])
]

# -----------------------------------------------------------------------------
# Set up date - TIME_STAMP_SHIFTED exists to make sure only data up to the
# PREVIOUS month is used.

expd["TIME_STAMP"] = [
    datetime.datetime(int(year), int(month), 1) for year, month in zip(expd["EXPNYR"], expd["EXPNMO"])
]

expd["TIME_STAMP_SHIFTED"] = [
    datetime.datetime(int(year), int(month), 15) for year, month in zip(expd["EXPNYR"], expd["EXPNMO"])
]

# -----------------------------------------------------------------------------
# Set up "BASKETID"

expd["BASKETID"] = [
    str(x) + "_" + y.strftime("%Y-%m") for x, y in zip(expd["NEWID"], expd["TIME_STAMP"])
]

# -----------------------------------------------------------------------------
# Build a training, validation and testing flag. We will use January to August
# for training, September and October for validation and November and December
# for testing. If you decide to add more data, you should probably come up
# with your own way of separating the data.

expd["Stage"] = [
    "Testing" if month > 10.0 else
    "Validation" if month > 8.0 else
    "Training" for month in expd["EXPNMO"]
]

# -----------------------------------------------------------------------------
# Set up UCCs - the UCCs are a way to systematically categorize products.
# Every digit has significance. That is why we create extra columns for
# that contain the first digit, the first two digits etc.

ucc = np.asarray(expd["UCC"]).astype(str)

expd["UCC1"] = [elem[:1] for elem in ucc]
expd["UCC2"] = [elem[:2] for elem in ucc]
expd["UCC3"] = [elem[:3] for elem in ucc]
expd["UCC4"] = [elem[:4] for elem in ucc]
expd["UCC5"] = [elem[:5] for elem in ucc]

# -----------------------------------------------------------------------------
# Set up units - this allows the engine to directly compare

units = dict()

units["UCC"] = "UCC"
units["UCC1"] = "UCC1"
units["UCC2"] = "UCC2"
units["UCC3"] = "UCC3"
units["UCC4"] = "UCC4"
units["UCC5"] = "UCC5"

# Adding 'comparison only' to the unit
# forces Multirel to always compare this
# column to others.
units["EXPNYR"] = "year, comparison only"

# -----------------------------------------------------------------------------
# Declare CATEGORICAL, DISCRETE, JOIN_KEYS, NUMERICAL, TARGETS, TIME_STAMPS

CATEGORICAL = [
    "UCC",
    "UCC1",
    "UCC2",
    "UCC3",
    "UCC4",
    "UCC5"
]

DISCRETE = [
    "EXPNYR"
]

JOIN_KEYS = [
    "NEWID",
    "BASKETID"
]

NUMERICAL = [
    "COST"
]

TARGETS = [
    "TARGET"
]

TIME_STAMPS = [
    "TIME_STAMP",
    "TIME_STAMP_SHIFTED"
]

# -----------------------------------------------------------------------------
# Prepare tables and store in folder. Only the population table needs to be split
# into a training, validation and testing set. The peripheral tables can be stored as whole.
# The condition t2.TIME_STAMP <= t1.TIME_STAMP will ensure that there are no
# easter eggs.

df_population_training = engine.DataFrame(
    "POPULATION_TRAINING",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS,
    units=units
).send(
    expd[expd["Stage"] == "Training"]
)

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
).send(
    expd[expd["Stage"] == "Validation"]
)

df_population_validation.save()

df_population_testing = engine.DataFrame(
    "POPULATION_TESTING",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS,
    units=units
).send(
    expd[expd["Stage"] == "Testing"]
)

df_population_testing.save()

# -----------------------------------------------------------------------------
# The peripheral table simply contains the entire dataset.

df_peripheral = engine.DataFrame(
    "PERIPHERAL",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS,
    units=units
).send(expd)

df_peripheral.save()

# -----------------------------------------------------------------------------
# Print time taken

end = time.time()

print("Time taken: " + str(end - begin) + " seconds.")
