import datetime
import os
import time

import numpy as np
import pandas as pd

import getml.data as data
import getml.engine as engine

# -----------------------------------------------------------------------------
# Set up folders - you need to insert folders on your computer

# The folder that contains expd151.csv
RAW_DATA_FOLDER = os.getenv("HOME") + "/Downloads/diary15"

# -----------------------------------------------------------------------------

engine.set_project("CE")

# -----------------------------------------------------------------------------
# Begin timing

begin = time.time()

# #############################################################################
# Load data.

os.chdir(RAW_DATA_FOLDER)

# -----------------------------------------------------------------------------
# Load EXPD

expd = pd.read_csv("expd151.csv")
expd = expd.append(pd.read_csv("expd152.csv"))
expd = expd.append(pd.read_csv("expd153.csv"))
expd = expd.append(pd.read_csv("expd154.csv"))

# -----------------------------------------------------------------------------
# Load FMLD

fmld = pd.read_csv("fmld151.csv")
fmld = fmld.append(pd.read_csv("fmld152.csv"))
fmld = fmld.append(pd.read_csv("fmld153.csv"))
fmld = fmld.append(pd.read_csv("fmld154.csv"))

# -----------------------------------------------------------------------------
# Load MEMD

memd = pd.read_csv("memd151.csv")
memd = fmld.append(pd.read_csv("memd152.csv"))
memd = fmld.append(pd.read_csv("memd153.csv"))
memd = fmld.append(pd.read_csv("memd154.csv"))

# #############################################################################
# Staging EXPD.

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
# Set up date.

expd["TIME_STAMP"] = [
    datetime.datetime(int(year), int(month), 1) for year, month in zip(expd["EXPNYR"], expd["EXPNMO"])
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
# Add time stamp to MEMD.

memd["TIME_STAMP"] = [pd.Timestamp("2015-01-01") for i in range(memd.shape[0])]

# -----------------------------------------------------------------------------
# Make the population table

FMLD_COLS = [
    "INC_RANK",
    "INC_RNK1",
    "INC_RNK2",
    "INC_RNK3",
    "INC_RNK4",
    "INC_RNK5",
    "INC_RNKM",
    "NEWID"
]

population_all = expd.merge(
    fmld[FMLD_COLS],
    on="NEWID"
)

# -----------------------------------------------------------------------------
# Load EXPD into the engine.

EXPD_CATEGORICAL = [
    "UCC",
    "UCC1",
    "UCC2",
    "UCC3",
    "UCC4",
    "UCC5"
]

EXPD_JOIN_KEYS = [
    "NEWID"
]

EXPD_NUMERICAL = [
    "COST",
    "EXPNYR",
    "EXPNMO"
]

EXPD_TARGETS = [
    "TARGET"
]

EXPD_TIME_STAMPS = [
    "TIME_STAMP"
]

expd_roles = {
    "join_key": EXPD_JOIN_KEYS,
    "time_stamp": EXPD_TIME_STAMPS,
    "categorical": EXPD_CATEGORICAL,
    "numerical": EXPD_NUMERICAL,
    "target": EXPD_TARGETS
}

df_expd = data.DataFrame.from_pandas(
    pandas_df=expd, 
    name="EXPD",
    roles=expd_roles,
    ignore=True)

df_expd.set_unit("UCC1", "UCC1")
df_expd.set_unit("UCC2", "UCC2")
df_expd.set_unit("UCC3", "UCC3")
df_expd.set_unit("UCC4", "UCC4")
df_expd.set_unit("UCC5", "UCC5")
df_expd.set_unit("UCC", "UCC")
df_expd.set_unit("EXPNMO", "month")

df_expd.save()

# -----------------------------------------------------------------------------
# Load POPULATION_ALL into the engine.

FMLD_NUMERICAL = [
    "INC_RANK",
    "INC_RNK1",
    "INC_RNK2",
    "INC_RNK3",
    "INC_RNK4",
    "INC_RNK5",
    "INC_RNKM"
]

population_roles = {
    "join_key": EXPD_JOIN_KEYS,
    "time_stamp": EXPD_TIME_STAMPS,
    "categorical": EXPD_CATEGORICAL,
    "numerical": EXPD_NUMERICAL + FMLD_NUMERICAL,
    "target": EXPD_TARGETS
}

df_population_all = data.DataFrame.from_pandas(
    pandas_df=population_all, 
    name="POPULATION_ALL",
    roles=population_roles,
    ignore=True)

df_population_all.set_unit("UCC1", "UCC1")
df_population_all.set_unit("UCC2", "UCC2")
df_population_all.set_unit("UCC3", "UCC3")
df_population_all.set_unit("UCC4", "UCC4")
df_population_all.set_unit("UCC5", "UCC5")
df_population_all.set_unit("UCC", "UCC")
df_population_all.set_unit("EXPNMO", "month")

df_population_all.save()

# -----------------------------------------------------------------------------
# Separate POPULATION_ALL into training, testing, validation set.

random = df_population_all.random()

df_population_training = df_population_all.where("POPULATION_TRAINING", random <= 0.7)

df_population_training.save()

df_population_validation = df_population_all.where("POPULATION_VALIDATION", (random <= 0.85) & (random > 0.7))

df_population_validation.save()

df_population_testing = df_population_all.where("POPULATION_TESTING", random > 0.85)

df_population_testing.save()

# -----------------------------------------------------------------------------
# Load MEMD

MEMD_CATEGORICAL = [
    "MARITAL",
    "SEX",
    "EMPLTYPE",
    "HISPANIC",
    "OCCULIST",
    "WHYNOWRK",
    "EDUCA",
    "MEDICARE",
    "PAYPERD",
    "RC_WHITE",
    "RC_BLACK",
    "RC_ASIAN",
    "RC_OTHER",
    "WKSTATUS"
]

MEMD_NUMERICAL = [
    "AGE",
    "WAGEX",
]

MEMD_JOIN_KEYS = [
    "NEWID"
]

MEMD_TIME_STAMPS = [
    "TIME_STAMP"
]

memd_roles = {
    "join_key": MEMD_JOIN_KEYS,
    "time_stamp": MEMD_TIME_STAMPS,
    "categorical": MEMD_CATEGORICAL,
    "numerical": MEMD_NUMERICAL
}

df_memd = data.DataFrame.from_pandas(
    pandas_df=memd, 
    name="MEMD",
    roles=memd_roles,
    ignore=True)

df_memd.save()

# -----------------------------------------------------------------------------
# Print time taken

end = time.time()

print("Time taken: " + str(end - begin) + " seconds.")
