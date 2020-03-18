import datetime
import os
import time

import getml.data as data 
import getml.engine as engine 
import getml.data.roles as roles 

# -----------------------------------------------------------------------------
# Set up folders - you need to insert folders on your computer

# The folder that contains expd151.csv
RAW_DATA_FOLDER = os.getenv("HOME") + "/Downloads/diary15/"

# -----------------------------------------------------------------------------

engine.set_project("CE")

# -----------------------------------------------------------------------------
# Begin timing

begin = time.time()

# #############################################################################
# Load data.

# -----------------------------------------------------------------------------
# Load EXPD

expd_fnames = [
    RAW_DATA_FOLDER + "expd151.csv",
    RAW_DATA_FOLDER + "expd152.csv",
    RAW_DATA_FOLDER + "expd153.csv",
    RAW_DATA_FOLDER + "expd154.csv"
]

expd_roles = {"unused_string": ["UCC", "NEWID"]}

df_expd = data.DataFrame.from_csv(
    fnames=expd_fnames,
    name="EXPD",
    roles=expd_roles
)

# -----------------------------------------------------------------------------
# Load FMLD

fmld_fnames = [
    RAW_DATA_FOLDER + "fmld151.csv",
    RAW_DATA_FOLDER + "fmld152.csv",
    RAW_DATA_FOLDER + "fmld153.csv",
    RAW_DATA_FOLDER + "fmld154.csv"
]

# The sniffer will interpret NEWID
# as a numeric column. But we want it
# to be treated as a string.
fmld_roles = {"unused_string": ["NEWID"]}

df_fmld = data.DataFrame.from_csv(
    fnames=fmld_fnames,
    name="FMLD",
    roles=fmld_roles
)

# -----------------------------------------------------------------------------
# Load MEMD

memd_fnames = [
    RAW_DATA_FOLDER + "memd151.csv",
    RAW_DATA_FOLDER + "memd152.csv",
    RAW_DATA_FOLDER + "memd153.csv",
    RAW_DATA_FOLDER + "memd154.csv"
]

# The sniffer will interpret NEWID
# as a numeric column. But we want it
# to be treated as a string.
memd_roles = {"unused_string": ["NEWID"]}

df_memd = data.DataFrame.from_csv(
    fnames=memd_fnames,
    name="MEMD",
    roles=memd_roles
)

# #############################################################################
# Staging EXPD.

# -----------------------------------------------------------------------------
# Make EXPNYR, EXPNMO and COST numerical columns

df_expd.set_role(["EXPNYR", "EXPNMO", "COST"], roles.numerical)

df_expd.set_unit(["EXPNMO"], "month")
df_expd.set_unit(["COST"], "cost")

# -----------------------------------------------------------------------------
# Make newid a join key.

df_expd.set_role("NEWID", roles.join_key)

# -----------------------------------------------------------------------------
# Remove all entries, for which EXPNYR or EXPNYR are nan.

expnyr = df_expd["EXPNYR"]
expnmo = df_expd["EXPNMO"]

not_nan = (expnyr.is_nan() | expnmo.is_nan()).is_false()

df_expd = df_expd.where("EXPD", not_nan)

# -----------------------------------------------------------------------------
# Generate time stamps.

expnyr = df_expd["EXPNYR"]
expnmo = df_expd["EXPNMO"]

ts = (expnyr.as_str() + "/" + expnmo.as_str()).as_ts(["%Y/%n"])

df_expd.add(ts, "TIME_STAMP", roles.time_stamp)

# -----------------------------------------------------------------------------
# Add UCC substrings

ucc = df_expd["UCC"]

for i in range(5):
    substr = ucc.substr(0, i+1)
    df_expd.add(
            substr, 
            name="UCC" + str(i+1),
            role=roles.categorical,
            unit="UCC" + str(i+1))

df_expd.set_role("UCC", roles.categorical)
df_expd.set_unit("UCC", "UCC")

# -----------------------------------------------------------------------------
# Add target
# "GIFT" is 1 when it is a gift and 2 otherwise.
# We want to change that to 1 and 0.

target = (df_expd["GIFT"] == 1)

df_expd.add(target, "TARGET", roles.target)

# -----------------------------------------------------------------------------

df_expd.save()

# #############################################################################
# Staging MEMD.

df_memd.set_role([
    "MARITAL",
    "SEX",
    "EMPLTYPE",
    "OCCULIST",
    "WHYNOWRK",
    "EDUCA",
    "MEDICARE",
    "PAYPERD",
    "RC_WHITE",
    "RC_BLACK",
    "RC_ASIAN",
    "RC_OTHER",
    "WKSTATUS"], roles.categorical)

df_memd.set_role(["AGE", "WAGEX"], roles.numerical)

df_memd.set_role("NEWID", roles.join_key)

time_stamp = df_memd.string_column("2015/01/01").as_ts(["%Y/%m/%d"])

df_memd.add(time_stamp, "TIME_STAMP", roles.time_stamp)

df_memd.save()

# #############################################################################
# Staging POPULATION.

# -----------------------------------------------------------------------------
# Separate EXPD in training, testing, validation set

random = df_expd.random()

df_population_training = df_expd.where("POPULATION_TRAINING", random <= 0.7)

df_population_validation = df_expd.where("POPULATION_VALIDATION", (random <= 0.85) & (random > 0.7))

df_population_testing = df_expd.where("POPULATION_TESTING", random > 0.85)

# -----------------------------------------------------------------------------------------------
# NEWID in FMLD is unique - therefore, we can just LEFT JOIN it onto the POPULATION tables. 

income_ranks = [
    "INC_RANK",
    "INC_RNK1",
    "INC_RNK2",
    "INC_RNK3",
    "INC_RNK4",
    "INC_RNK5",
    "INC_RNKM"
]

df_fmld.set_role(income_ranks, roles.numerical)

for inc in income_ranks:
    df_fmld.set_unit(inc, inc)

df_fmld.set_role("NEWID", roles.join_key)

df_population_training = df_population_training.join(
        name="POPULATION_TRAINING", 
        other=df_fmld, 
        join_key="NEWID",
        other_cols=[
            df_fmld["INC_RANK"],
            df_fmld["INC_RNK1"],
            df_fmld["INC_RNK2"],
            df_fmld["INC_RNK3"],
            df_fmld["INC_RNK4"],
            df_fmld["INC_RNK5"],
            df_fmld["INC_RNKM"]
        ]
)

df_population_validation = df_population_validation.join(
        name="POPULATION_VALIDATION", 
        other=df_fmld, 
        join_key="NEWID",
        other_cols=[
            df_fmld["INC_RANK"],
            df_fmld["INC_RNK1"],
            df_fmld["INC_RNK2"],
            df_fmld["INC_RNK3"],
            df_fmld["INC_RNK4"],
            df_fmld["INC_RNK5"],
            df_fmld["INC_RNKM"]
        ]
)

df_population_testing = df_population_testing.join(
        name="POPULATION_TESTING", 
        other=df_fmld, 
        join_key="NEWID",
        other_cols=[
            df_fmld["INC_RANK"],
            df_fmld["INC_RNK1"],
            df_fmld["INC_RNK2"],
            df_fmld["INC_RNK3"],
            df_fmld["INC_RNK4"],
            df_fmld["INC_RNK5"],
            df_fmld["INC_RNKM"]
        ]
)

# -----------------------------------------------------------------------------------------------

df_population_training.save()

df_population_validation.save()

df_population_testing.save()

# #############################################################################
# Print time taken

end = time.time()

print("Time taken: " + str(end - begin) + " seconds.")

