import datetime
import os
import time

import getml.engine as engine

# -----------------------------------------------------------------------------

engine.set_project("CE")

# -----------------------------------------------------------------------------
# Begin timing

begin = time.time()

# -----------------------------------------------------------------------------
# Define the source files the source files
source_path = os.path.join(os.getcwd(), "../../../data/consumer_expenditures/raw/")

csv_fnames = [
    os.path.join(source_path, "expd151.csv"),
    os.path.join(source_path, "expd152.csv"),
    os.path.join(source_path, "expd153.csv"),
    os.path.join(source_path, "expd154.csv")
]

# -----------------------------------------------------------------------------
# Declare CATEGORICAL, DISCRETE, JOIN_KEYS, NUMERICAL, TARGETS, TIME_STAMPS

CATEGORICAL = [
    "UCC"
]

DISCRETE = [
    "EXPNYR",
    "EXPNMO",
    "GIFT"
]

JOIN_KEYS = [
    "NEWID"
]

NUMERICAL = [
    "COST"
]

TARGETS = []

TIME_STAMPS = []

# -----------------------------------------------------------------------------

df_peripheral = engine.DataFrame(
    "PERIPHERAL",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS
)

df_peripheral.read_csv(fnames=csv_fnames, append=False)

# -----------------------------------------------------------------------------
# Remove all entries, for which EXPNYR or EXPNYR are nan.

expnyr = df_peripheral.discrete("EXPNYR")
expnmo = df_peripheral.discrete("EXPNMO")

not_nan = (expnyr.is_nan() | expnmo.is_nan()).is_false()

df_peripheral = df_peripheral.where("PERIPHERAL", not_nan)

# -----------------------------------------------------------------------------
# Set units for expnyr and expnmo

expnyr.set_unit("year, comparison only")
expnmo.set_unit("month, comparison only")

# -----------------------------------------------------------------------------
# Generate basketid.

newid = df_peripheral.join_key("NEWID")

basketid = newid + "_" + expnyr.to_str() + "-" + expnmo.to_str()

df_peripheral.add_join_key(basketid, "BASKETID")

# -----------------------------------------------------------------------------
# Generate time stamps.

ts = (expnyr.to_str() + "/" + expnmo.to_str()).to_ts(["%Y/%n"])

df_peripheral.add_time_stamp(ts, "TIME_STAMP")

ts = df_peripheral.time_stamp("TIME_STAMP")

df_peripheral.add_time_stamp(ts + 15.0, "TIME_STAMP_SHIFTED")

# -----------------------------------------------------------------------------
# Add UCC substrings

ucc = df_peripheral.categorical("UCC")

ucc.set_unit("UCC")

for i in range(5):
    substr = ucc.substr(0, i+1)
    df_peripheral.add_categorical(substr, "UCC" + str(i+1), "UCC" + str(i+1))

# -----------------------------------------------------------------------------
# Add target

gift = df_peripheral.discrete("GIFT")

# "GIFT" is 1 when it is a gift and 2 otherwise.
# We want to change that to 1 and 0.
target = gift.update(gift==2.0, 0.0)

df_peripheral.add_target(target, "TARGET")

df_peripheral.rm_discrete("GIFT")

# -----------------------------------------------------------------------------
# Separate in training, testing, validation set

df_population_training = df_peripheral.where("POPULATION_TRAINING", expnmo <= 8)

df_population_validation = df_peripheral.where("POPULATION_VALIDATION", (expnmo <= 10) & (expnmo > 8))

df_population_testing = df_peripheral.where("POPULATION_TESTING", expnmo > 10)

# -----------------------------------------------------------------------------
# Save

df_peripheral.save()

df_population_training.save()

df_population_validation.save()

df_population_testing.save()

# -----------------------------------------------------------------------------
# Print time taken

end = time.time()

print("Time taken: " + str(end - begin) + " seconds.")
