## This script imports the CE data and formats it in such a way the
## user can handle it way more easily in the getting started guide.

import datetime
import os

import numpy as np
import pandas as pd

## -------------------------------------------------------------------
## Setup

# The folder that contains all required .csv files.
RAW_DATA_FOLDER = "./"

## -------------------------------------------------------------------

## Read the data from the source files
expd = pd.read_csv(os.path.join(RAW_DATA_FOLDER, "expd151.csv"))
expd = expd.append(pd.read_csv(os.path.join(RAW_DATA_FOLDER, "expd152.csv")))
expd = expd.append(pd.read_csv(os.path.join(RAW_DATA_FOLDER, "expd153.csv")))
expd = expd.append(pd.read_csv(os.path.join(RAW_DATA_FOLDER, "expd154.csv")))

# -----------------------------------------------------------------------------
# Set up target - we want to predict whether the item is a gift

expd["TARGET"] = [0.0 if elem == 2 else 1.0 for elem in expd["GIFT"]]

# -----------------------------------------------------------------------------
# Remove the instances where date is nan - they will be ignored by the AutoSQL
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

## -------------------------------------------------------------------

## Export data into new .csv files.
expd[expd["Stage"] == "Training"].to_csv("../CE_population_training.csv")
expd[expd["Stage"] == "Validation"].to_csv("../CE_population_validation.csv")
expd.to_csv("../CE_peripheral.csv")
