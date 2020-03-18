import datetime
import os
import time

import numpy as np

import getml.data as data
import getml.database as database
import getml.engine as engine

# -----------------------------------------------------------------------------
# Set up the MySQL connection.

database.connect_mysql(
    host="localhost",
    port=3306,
    dbname="mydb",
    user="myuser",
    password="mypassword",
    time_formats=['%Y/%m/%d']
)

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

query = database.sniff_csv("EXPD_RAW", expd_fnames)

print(query)

# The sniffer will interpret UCC and NEWID
# as DOUBLE columns. But we want them
# to be treated as TEXT.
database.execute(""" 
DROP TABLE IF EXISTS EXPD_RAW;

CREATE TABLE EXPD_RAW(
    NEWID    TEXT,
    ALLOC    INT,
    COST     DOUBLE,
    GIFT     INT,
    PUB_FLAG INT,
    UCC      TEXT,
    EXPNSQDY TEXT,
    EXPN_QDY TEXT,
    EXPNWKDY TEXT,
    EXPN_KDY TEXT,
    EXPNMO   TEXT,
    EXPNMO_  TEXT,
    EXPNYR   TEXT,
    EXPNYR_  TEXT);
""")

database.read_csv("EXPD_RAW", expd_fnames)


# -----------------------------------------------------------------------------
# Load FMLD

# FMLD's original row size is too large for MySQL, so we have to use our own data frames
# to load the data.

fmld_fnames = [
    RAW_DATA_FOLDER + "fmld151.csv",
    RAW_DATA_FOLDER + "fmld152.csv",
    RAW_DATA_FOLDER + "fmld153.csv",
    RAW_DATA_FOLDER + "fmld154.csv"
]

FMLD_NUMERICAL = [
    "INC_RANK",
    "INC_RNK1",
    "INC_RNK2",
    "INC_RNK3",
    "INC_RNK4",
    "INC_RNK5",
    "INC_RNKM"
]

FMLD_JOIN_KEYS = [
    "NEWID"
]

fmld_roles = {
    "join_key": FMLD_JOIN_KEYS,
    "numerical": FMLD_NUMERICAL
}

df_fmld = data.DataFrame.from_csv(
    fnames=fmld_fnames,
    name="FMLD",
    roles=fmld_roles,
    ignore=True
)

df_fmld.to_db("FMLD")

# -----------------------------------------------------------------------------
# Load MEMD

memd_fnames = [
    RAW_DATA_FOLDER + "memd151.csv",
    RAW_DATA_FOLDER + "memd152.csv",
    RAW_DATA_FOLDER + "memd153.csv",
    RAW_DATA_FOLDER + "memd154.csv"
]

query = database.sniff_csv("MEMD_RAW", memd_fnames)

print(query)

database.execute("""
DROP TABLE IF EXISTS MEMD_RAW;

CREATE TABLE MEMD_RAW(
    OCCULIST TEXT,
    HRSPERWK TEXT,
    WKS_WRKD TEXT,
    EMPLTYPE TEXT,
    MARITAL  INT,
    HISPANIC TEXT,
    WHYNOWRK TEXT,
    MEMBRACE INT,
    SEX      INT,
    HRSP_RWK TEXT,
    WKS__RKD TEXT,
    EMPL_YPE TEXT,
    HISP_NIC TEXT,
    WHYN_WRK TEXT,
    OCCU_IST TEXT,
    NEWID    TEXT,
    AGE      INT,
    AGE_     TEXT,
    ANGVX    TEXT,
    ANGVX_   TEXT,
    ANPVTX   TEXT,
    ANPVTX_  TEXT,
    ANRRX    TEXT,
    ANRRX_   TEXT,
    CU_CODE1 INT,
    EDUCA    TEXT,
    EDUCA_   TEXT,
    GROSPAYX TEXT,
    GROS_AYX TEXT,
    GVX      TEXT,
    GVX_     TEXT,
    IRAX     TEXT,
    IRAX_    TEXT,
    JSSDEDX  TEXT,
    JSSDEDX_ TEXT,
    MEMBNO   INT,
    PVTX     TEXT,
    PVTX_    TEXT,
    RRX      TEXT,
    RRX_     TEXT,
    SCHLNCHQ TEXT,
    SCHL_CHQ TEXT,
    SCHLNCHX TEXT,
    SCHL_CHX TEXT,
    SLFEMPSS TEXT,
    SLFE_PSS TEXT,
    SS_RRX   TEXT,
    SS_RRX_  TEXT,
    SUPPX    TEXT,
    SUPPX_   TEXT,
    US_SUPP  TEXT,
    US_SUPP_ TEXT,
    WAGEX    TEXT,
    WAGEX_   TEXT,
    SS_RRQ   TEXT,
    SS_RRQ_  TEXT,
    SOCRRX   TEXT,
    SOCRRX_  TEXT,
    ARM_FORC TEXT,
    ARM__ORC TEXT,
    IN_COLL  TEXT,
    IN_COLL_ TEXT,
    MEDICARE TEXT,
    MEDI_ARE TEXT,
    PAYPERD  TEXT,
    PAYPERD_ TEXT,
    HORIGIN  INT,
    RC_WHITE TEXT,
    RC_W_ITE TEXT,
    RC_BLACK TEXT,
    RC_B_ACK TEXT,
    RC_NATAM TEXT,
    RC_N_TAM TEXT,
    RC_ASIAN TEXT,
    RC_A_IAN TEXT,
    RC_PACIL TEXT,
    RC_P_CIL TEXT,
    RC_OTHER TEXT,
    RC_O_HER TEXT,
    RC_DK    TEXT,
    RC_DK_   TEXT,
    ANGVXM   TEXT,
    ANGVXM_  TEXT,
    ANPVTXM  TEXT,
    ANPVTXM_ TEXT,
    ANRRXM   TEXT,
    ANRRXM_  TEXT,
    JSSDEDXM TEXT,
    JSSD_DXM TEXT,
    JSSDEDX1 TEXT,
    JSSDEDX2 TEXT,
    JSSDEDX3 TEXT,
    JSSDEDX4 TEXT,
    JSSDEDX5 TEXT,
    SLFEMPSM TEXT,
    SLFE_PSM TEXT,
    SLFEMPS1 TEXT,
    SLFEMPS2 TEXT,
    SLFEMPS3 TEXT,
    SLFEMPS4 TEXT,
    SLFEMPS5 TEXT,
    SOCRRXM  TEXT,
    SOCRRXM_ TEXT,
    SOCRRX1  TEXT,
    SOCRRX2  TEXT,
    SOCRRX3  TEXT,
    SOCRRX4  TEXT,
    SOCRRX5  TEXT,
    SS_RRXM  TEXT,
    SS_RRXM_ TEXT,
    SS_RRX1  TEXT,
    SS_RRX2  TEXT,
    SS_RRX3  TEXT,
    SS_RRX4  TEXT,
    SS_RRX5  TEXT,
    SS_RRXI  TEXT,
    SUPPXM   TEXT,
    SUPPXM_  TEXT,
    SUPPX1   TEXT,
    SUPPX2   TEXT,
    SUPPX3   TEXT,
    SUPPX4   TEXT,
    SUPPX5   TEXT,
    SUPPXI   TEXT,
    WAGEXM   TEXT,
    WAGEXM_  TEXT,
    WAGEX1   TEXT,
    WAGEX2   TEXT,
    WAGEX3   TEXT,
    WAGEX4   TEXT,
    WAGEX5   TEXT,
    WAGEXI   TEXT,
    SS_RRB   TEXT,
    SS_RRB_  TEXT,
    SS_RRBX  TEXT,
    SS_RRBX_ TEXT,
    SUPPB    TEXT,
    SUPPB_   TEXT,
    SUPPBX   TEXT,
    SUPPBX_  TEXT,
    WAGEB    TEXT,
    WAGEB_   TEXT,
    WAGEBX   TEXT,
    WAGEBX_  TEXT,
    ASIAN    TEXT,
    ASIAN_   TEXT,
    OCCUEARN TEXT,
    PAYSTUB  TEXT,
    PAYSTUB_ TEXT,
    SEMPFRM  TEXT,
    SEMPFRM_ TEXT,
    SEMPFRMX TEXT,
    SEMP_RMX TEXT,
    SMPFRMB  TEXT,
    SMPFRMB_ TEXT,
    SMPFRMBX TEXT,
    SMPF_MBX TEXT,
    SEMPFRM1 TEXT,
    SEMPFRM2 TEXT,
    SEMPFRM3 TEXT,
    SEMPFRM4 TEXT,
    SEMPFRM5 TEXT,
    SEMPFRMI TEXT,
    SEMPFRMM TEXT,
    SEMP_RMM TEXT,
    SOCSRRET TEXT,
    SOCS_RET TEXT,
    WKSTATUS TEXT);
""")

database.read_csv("MEMD_RAW", memd_fnames)

# -----------------------------------------------------------------------------
# Preprocess EXPD.

database.execute("""
    DROP TABLE IF EXISTS EXPD;

    CREATE TABLE EXPD AS
    SELECT CASE WHEN GIFT=2 THEN 0 ELSE 1 END AS TARGET,
           CONCAT(EXPNYR, "/", EXPNMO, "/", "01") AS TIME_STAMP,
           NEWID,
           EXPNYR,
           CAST(EXPNMO AS UNSIGNED) AS EXPNMO,
           COST,
           SUBSTR(UCC, 1, 1) AS UCC1,
           SUBSTR(UCC, 1, 2) AS UCC2,
           SUBSTR(UCC, 1, 3) AS UCC3,
           SUBSTR(UCC, 1, 4) AS UCC4,
           SUBSTR(UCC, 1, 5) AS UCC5,
           SUBSTR(UCC, 1, 6) AS UCC
    FROM EXPD_RAW
    WHERE EXPNMO != '';
""")

# -----------------------------------------------------------------------------
# Preprocess MEMD.

database.execute("""
    DROP TABLE IF EXISTS MEMD;

    CREATE TABLE MEMD AS
    SELECT MARITAL,
           SEX,
           EMPLTYPE,
           HISPANIC,
           OCCULIST,
           WHYNOWRK,
           EDUCA,
           MEDICARE,
           PAYPERD,
           RC_WHITE,
           RC_BLACK,
           RC_ASIAN,
           RC_OTHER,
           WKSTATUS,
           AGE,
           WAGEX,
           NEWID,
           '2015/01/01' AS TIME_STAMP
    FROM MEMD_RAW
""")

# -----------------------------------------------------------------------------
# Make POPULATION TABLE

database.execute("""
    CREATE INDEX EXPD_INDEX ON EXPD(NEWID(10)); 
    CREATE INDEX FMLD_INDEX ON FMLD(NEWID(10)); 

    DROP TABLE IF EXISTS POPULATION_ALL;

    CREATE TABLE POPULATION_ALL AS 
    SELECT t1.*,
           t2.INC_RANK,
           t2.INC_RNK1,
           t2.INC_RNK2,
           t2.INC_RNK3,
           t2.INC_RNK4,
           t2.INC_RNK5,
           t2.INC_RNKM
    FROM EXPD t1
    LEFT JOIN FMLD t2
    ON t1.NEWID = t2.NEWID;
""")

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

df_expd = data.DataFrame.from_db(
    table_name="EXPD", 
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

df_population_all = data.DataFrame.from_db(
    table_name="POPULATION_ALL", 
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

df_memd = data.DataFrame.from_db(
    table_name="MEMD", 
    name="MEMD",
    roles=memd_roles,
    ignore=True)

df_memd.save()

# -----------------------------------------------------------------------------
# Print time taken

end = time.time()

print("Time taken: " + str(end - begin) + " seconds.")

