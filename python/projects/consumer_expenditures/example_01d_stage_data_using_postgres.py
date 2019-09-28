import datetime
import os
import time

import numpy as np

import getml.database as database
import getml.engine as engine

# -----------------------------------------------------------------------------

engine.set_project("CE")

# -----------------------------------------------------------------------------
# Begin timing

begin = time.time()

# -----------------------------------------------------------------------------
# Build up the connection to postgres.
# Here, we are assuming that your PostgreSQL instance is running on the same
# computer as the one hosting the get.ML engine.
#
# If you are unsure what port your PostgreSQL instance is running on, call
# SELECT * FROM pg_settings WHERE name = 'port'; in psql.
#
# To get you started quickly, here is how to create a database and a user in
# psql and then connect to it:
# CREATE DATABASE mydb;
# CREATE USER myuser WITH ENCRYPTED PASSWORD 'mypassword';
# GRANT ALL PRIVILEGES ON DATABASE mydb TO myuser;
# \connect mydb;


database.connect_postgres(
    pg_hostaddr="127.0.0.1",
    pg_host="localhost",
    pg_port=5432,
    dbname="mydb",
    user="myuser",
    password="mypassword",
    time_formats=['%Y/%m/%d']
)

# -----------------------------------------------------------------------------
# Read the data from the source files. After you have executed these lines, you
# should already by able to see the table in psql as well as in the get.ML
# monitor.
source_path = os.path.join(os.getcwd(), "../../../data/consumer_expenditures/raw/")

csv_fnames = [
    os.path.join(source_path, "expd151.csv"),
    os.path.join(source_path, "expd152.csv"),
    os.path.join(source_path, "expd153.csv"),
    os.path.join(source_path, "expd154.csv")
]

query = database.sniff_csv("EXPD_RAW", csv_fnames)

database.execute(query)

database.read_csv("EXPD_RAW", csv_fnames)

# -----------------------------------------------------------------------------
# Do the preprocessing - note that names in PostgreSQL in always a good idea to
# put names of tables and columns into quotation marks.

database.execute("""
    DROP TABLE IF EXISTS "EXPD_ALL";

    CREATE TABLE "EXPD_ALL" AS
    SELECT CASE WHEN "GIFT"=2 THEN 0 ELSE 1 END AS "TARGET",
           "EXPNYR" || '/' || "EXPNMO" || '/' || '01' AS "TIME_STAMP",
           "EXPNYR" || '/' || "EXPNMO" || '/' || '15' AS "TIME_STAMP_SHIFTED",
           CAST("NEWID" AS INT) || '_' || "EXPNYR" || '-' || "EXPNMO" AS "BASKETID",
           CAST("NEWID" AS INT) AS "NEWID",
           "EXPNYR",
           CAST("EXPNMO" AS INT) AS "EXPNMO",
           "COST",
           substr(CAST("UCC" AS TEXT), 1, 1) AS "UCC1",
           substr(CAST("UCC" AS TEXT), 1, 2) AS "UCC2",
           substr(CAST("UCC" AS TEXT), 1, 3) AS "UCC3",
           substr(CAST("UCC" AS TEXT), 1, 4) AS "UCC4",
           substr(CAST("UCC" AS TEXT), 1, 5) AS "UCC5",
           substr(CAST("UCC" AS TEXT), 1, 6) AS "UCC"
    FROM "EXPD_RAW"
    WHERE "EXPNMO" != '';
""")

# -----------------------------------------------------------------------------
# Separate in training, testing and validation set.

database.execute("""
    DROP TABLE IF EXISTS "POPULATION_TRAINING";

    CREATE TABLE "POPULATION_TRAINING" AS
    SELECT "TARGET",
           "TIME_STAMP",
           "TIME_STAMP_SHIFTED",
           "BASKETID",
           "NEWID",
           "EXPNYR",
           "EXPNMO",
           "COST",
           "UCC1",
           "UCC2",
           "UCC3",
           "UCC4",
           "UCC5",
           "UCC"
    FROM "EXPD_ALL"
    WHERE "EXPNMO" <= 8;
""")

database.execute("""
    DROP TABLE IF EXISTS "POPULATION_VALIDATION";

    CREATE TABLE "POPULATION_VALIDATION" AS
    SELECT "TARGET",
           "TIME_STAMP",
           "TIME_STAMP_SHIFTED",
           "BASKETID",
           "NEWID",
           "EXPNYR",
           "EXPNMO",
           "COST",
           "UCC1",
           "UCC2",
           "UCC3",
           "UCC4",
           "UCC5",
           "UCC"
    FROM "EXPD_ALL"
    WHERE "EXPNMO" <= 10
    AND "EXPNMO" > 8;
""")

database.execute("""
    DROP TABLE IF EXISTS "POPULATION_TESTING";

    CREATE TABLE "POPULATION_TESTING" AS
    SELECT "TARGET",
           "TIME_STAMP",
           "TIME_STAMP_SHIFTED",
           "BASKETID",
           "NEWID",
           "EXPNYR",
           "EXPNMO",
           "COST",
           "UCC1",
           "UCC2",
           "UCC3",
           "UCC4",
           "UCC5",
           "UCC"
    FROM "EXPD_ALL"
    WHERE "EXPNMO" > 10;
""")

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
# We prepare a little helper function to set up the units.

def make_units(df):
  """Little helper function for setting up the units"""
  df.categorical("UCC1").set_unit("UCC1")
  df.categorical("UCC2").set_unit("UCC2")
  df.categorical("UCC3").set_unit("UCC3")
  df.categorical("UCC4").set_unit("UCC4")
  df.categorical("UCC5").set_unit("UCC5")
  df.categorical("UCC").set_unit("UCC")
  df.discrete("EXPNYR").set_unit("year, comparison only")

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
    targets=TARGETS
).from_db("POPULATION_TRAINING")

make_units(df_population_training)

df_population_training.save()

df_population_validation = engine.DataFrame(
    "POPULATION_VALIDATION",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS
).from_db("POPULATION_VALIDATION")

make_units(df_population_validation)

df_population_validation.save()

df_population_testing = engine.DataFrame(
    "POPULATION_TESTING",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS
).from_db("POPULATION_TESTING")

make_units(df_population_testing)

df_population_testing.save()

# -----------------------------------------------------------------------------
# The peripheral table simply contains the entire dataset. Note that
# DataFrame.from_db(...) can append to an existing data frame.
# (Obviously we could have also just read in EXPD_ALL.)

df_peripheral = engine.DataFrame(
    "PERIPHERAL",
    join_keys=JOIN_KEYS,
    time_stamps=TIME_STAMPS,
    categorical=CATEGORICAL,
    discrete=DISCRETE,
    numerical=NUMERICAL,
    targets=TARGETS
).from_db("POPULATION_TRAINING")

df_peripheral.from_db("POPULATION_VALIDATION", append=True)

df_peripheral.from_db("POPULATION_TESTING", append=True)

make_units(df_peripheral)

df_peripheral.save()

# -----------------------------------------------------------------------------
# Print time taken

end = time.time()

print("Time taken: " + str(end - begin) + " seconds.")
