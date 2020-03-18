import datetime
import os
import time

import numpy as np

import getml.data as data
import getml.database as database
import getml.engine as engine

# -----------------------------------------------------------------------------
# Set up folders - you need to insert folders on your computer

# The folder that contains expd151.csv
RAW_DATA_FOLDER = os.getenv("HOME") + "/Downloads/diary15/"

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
    hostaddr="127.0.0.1",
    host="localhost",
    port=5432,
    dbname="mydb",
    user="myuser",
    password="mypassword",
    time_formats=['%Y/%m/%d']
)

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
# as DOUBLE PRECISION columns. But we want them
# to be treated as TEXT.
database.execute(""" 
DROP TABLE IF EXISTS "EXPD_RAW";

CREATE TABLE "EXPD_RAW"(
    "NEWID"    TEXT,
    "ALLOC"    INTEGER,
    "COST"     DOUBLE PRECISION,
    "GIFT"     INTEGER,
    "PUB_FLAG" INTEGER,
    "UCC"      TEXT,
    "EXPNSQDY" TEXT,
    "EXPN_QDY" TEXT,
    "EXPNWKDY" TEXT,
    "EXPN_KDY" TEXT,
    "EXPNMO"   TEXT,
    "EXPNMO_"  TEXT,
    "EXPNYR"   TEXT,
    "EXPNYR_"  TEXT);
""")

database.read_csv("EXPD_RAW", expd_fnames)

# -----------------------------------------------------------------------------
# Load FMLD

fmld_fnames = [
    RAW_DATA_FOLDER + "fmld151.csv",
    RAW_DATA_FOLDER + "fmld152.csv",
    RAW_DATA_FOLDER + "fmld153.csv",
    RAW_DATA_FOLDER + "fmld154.csv"
]

query = database.sniff_csv("FMLD_RAW", fmld_fnames)

print(query)

# The sniffer will interpret NEWID as
# as a DOUBLE PRECISION column. But we want it
# to be treated as TEXT.
# We can just copy and paste the original
# query suggested by the sniffer and
# make the modifications we want.
database.execute(""" 
DROP TABLE IF EXISTS "FMLD_RAW";

CREATE TABLE "FMLD_RAW"(
    "INC_RNKM" DOUBLE PRECISION,
    "INC_RNK5" DOUBLE PRECISION,
    "INC_RNK4" DOUBLE PRECISION,
    "INC_RNK3" DOUBLE PRECISION,
    "INC_RNK2" DOUBLE PRECISION,
    "INC_RNK1" DOUBLE PRECISION,
    "INC_RANK" DOUBLE PRECISION,
    "NEWID"    TEXT,
    "AGE_REF"  INTEGER,
    "AGE_REF_" TEXT,
    "AGE2"     TEXT,
    "AGE2_"    TEXT,
    "BLS_URBN" INTEGER,
    "CUTENURE" INTEGER,
    "CUTE_URE" TEXT,
    "DESCRIP"  TEXT,
    "DESCRIP_" TEXT,
    "EARNCOMP" INTEGER,
    "EARN_OMP" TEXT,
    "EDUC_REF" DOUBLE PRECISION,
    "EDUC0REF" TEXT,
    "EDUCA2"   TEXT,
    "EDUCA2_"  TEXT,
    "EMPLTYP1" TEXT,
    "EMPL_YP1" TEXT,
    "EMPLTYP2" TEXT,
    "EMPL_YP2" TEXT,
    "FAM_SIZE" INTEGER,
    "FAM__IZE" TEXT,
    "FAM_TYPE" INTEGER,
    "FAM__YPE" TEXT,
    "FGVX"     INTEGER,
    "FGVX_"    TEXT,
    "FINCBEFX" INTEGER,
    "FINC_EFX" TEXT,
    "FINLWT21" DOUBLE PRECISION,
    "FIRAX"    INTEGER,
    "FIRAX_"   TEXT,
    "FJSSDEDX" INTEGER,
    "FJSS_EDX" TEXT,
    "FPVTX"    INTEGER,
    "FPVTX_"   TEXT,
    "FREEMLX"  TEXT,
    "FREEMLX_" TEXT,
    "FRRX"     INTEGER,
    "FRRX_"    TEXT,
    "FS_MTHI"  TEXT,
    "FS_MTHI_" TEXT,
    "FSS_RRX"  INTEGER,
    "FSS_RRX_" TEXT,
    "FSUPPX"   INTEGER,
    "FSUPPX_"  TEXT,
    "FWAGEX"   INTEGER,
    "FWAGEX_"  TEXT,
    "HRSPRWK1" TEXT,
    "HRSP_WK1" TEXT,
    "HRSPRWK2" TEXT,
    "HRSP_WK2" TEXT,
    "JFS_AMT"  INTEGER,
    "JFS_AMT_" TEXT,
    "JGRCFDMV" TEXT,
    "JGRC_DMV" TEXT,
    "JGRCFDWK" TEXT,
    "JGRC_DWK" TEXT,
    "JGROCYMV" TEXT,
    "JGRO_YMV" TEXT,
    "JGROCYWK" TEXT,
    "JGRO_YWK" TEXT,
    "LUMPX"    TEXT,
    "LUMPX_"   TEXT,
    "MARITAL1" INTEGER,
    "MARI_AL1" TEXT,
    "NO_EARNR" INTEGER,
    "NO_E_RNR" TEXT,
    "OCCEXPNX" TEXT,
    "OCCE_PNX" TEXT,
    "OCCULIS2" TEXT,
    "OCCU_IS2" TEXT,
    "OTHINX"   TEXT,
    "OTHINX_"  TEXT,
    "OTHRECX"  INTEGER,
    "OTHRECX_" TEXT,
    "PERSLT18" INTEGER,
    "PERS_T18" TEXT,
    "PERSOT64" INTEGER,
    "PERS_T64" TEXT,
    "OCCULIS1" TEXT,
    "OCCU_IS1" TEXT,
    "POPSIZE"  INTEGER,
    "RACE2"    TEXT,
    "RACE2_"   TEXT,
    "REC_FS"   TEXT,
    "REC_FS_"  TEXT,
    "REF_RACE" INTEGER,
    "REF__ACE" TEXT,
    "REGION"   TEXT,
    "SEX_REF"  INTEGER,
    "SEX_REF_" TEXT,
    "SEX2"     TEXT,
    "SEX2_"    TEXT,
    "SMSASTAT" INTEGER,
    "STRTMNTH" DOUBLE PRECISION,
    "STRTYEAR" INTEGER,
    "TYPOWND"  TEXT,
    "TYPOWND_" TEXT,
    "VEHQ"     TEXT,
    "VEHQ_"    TEXT,
    "WEEKI"    INTEGER,
    "WEEKI_"   TEXT,
    "WEEKN"    INTEGER,
    "WELFRX"   TEXT,
    "WELFRX_"  TEXT,
    "WHYNWRK1" TEXT,
    "WHYN_RK1" TEXT,
    "WHYNWRK2" TEXT,
    "WHYN_RK2" TEXT,
    "WK_WRKD1" INTEGER,
    "WK_W_KD1" TEXT,
    "WK_WRKD2" TEXT,
    "WK_W_KD2" TEXT,
    "WTREP01"  TEXT,
    "WTREP02"  TEXT,
    "WTREP03"  TEXT,
    "WTREP04"  TEXT,
    "WTREP05"  TEXT,
    "WTREP06"  TEXT,
    "WTREP07"  TEXT,
    "WTREP08"  TEXT,
    "WTREP09"  TEXT,
    "WTREP10"  TEXT,
    "WTREP11"  TEXT,
    "WTREP12"  TEXT,
    "WTREP13"  TEXT,
    "WTREP14"  TEXT,
    "WTREP15"  TEXT,
    "WTREP16"  TEXT,
    "WTREP17"  TEXT,
    "WTREP18"  TEXT,
    "WTREP19"  TEXT,
    "WTREP20"  TEXT,
    "WTREP21"  TEXT,
    "WTREP22"  TEXT,
    "WTREP23"  TEXT,
    "WTREP24"  TEXT,
    "WTREP25"  TEXT,
    "WTREP26"  TEXT,
    "WTREP27"  TEXT,
    "WTREP28"  TEXT,
    "WTREP29"  TEXT,
    "WTREP30"  TEXT,
    "WTREP31"  TEXT,
    "WTREP32"  TEXT,
    "WTREP33"  TEXT,
    "WTREP34"  TEXT,
    "WTREP35"  TEXT,
    "WTREP36"  TEXT,
    "WTREP37"  TEXT,
    "WTREP38"  TEXT,
    "WTREP39"  TEXT,
    "WTREP40"  TEXT,
    "WTREP41"  TEXT,
    "WTREP42"  TEXT,
    "WTREP43"  TEXT,
    "WTREP44"  TEXT,
    "FOODTOT"  DOUBLE PRECISION,
    "FOODHOME" DOUBLE PRECISION,
    "CEREAL"   DOUBLE PRECISION,
    "BAKEPROD" DOUBLE PRECISION,
    "BEEF"     DOUBLE PRECISION,
    "PORK"     DOUBLE PRECISION,
    "OTHMEAT"  DOUBLE PRECISION,
    "POULTRY"  DOUBLE PRECISION,
    "SEAFOOD"  DOUBLE PRECISION,
    "EGGS"     DOUBLE PRECISION,
    "MILKPROD" DOUBLE PRECISION,
    "OTHDAIRY" DOUBLE PRECISION,
    "FRSHFRUT" DOUBLE PRECISION,
    "FRSHVEG"  DOUBLE PRECISION,
    "PROCFRUT" DOUBLE PRECISION,
    "PROCVEG"  DOUBLE PRECISION,
    "SWEETS"   DOUBLE PRECISION,
    "NONALBEV" DOUBLE PRECISION,
    "OILS"     DOUBLE PRECISION,
    "MISCFOOD" DOUBLE PRECISION,
    "FOODAWAY" DOUBLE PRECISION,
    "ALCBEV"   DOUBLE PRECISION,
    "SMOKSUPP" DOUBLE PRECISION,
    "PET_FOOD" DOUBLE PRECISION,
    "PERSPROD" DOUBLE PRECISION,
    "PERSSERV" DOUBLE PRECISION,
    "DRUGSUPP" DOUBLE PRECISION,
    "HOUSKEEP" DOUBLE PRECISION,
    "HH_CU_Q"  INTEGER,
    "HH_CU_Q_" TEXT,
    "HHID"     TEXT,
    "HHID_"    TEXT,
    "CHILDAGE" INTEGER,
    "CHIL_AGE" TEXT,
    "INCLASS"  DOUBLE PRECISION,
    "STATE"    TEXT,
    "INC__ANK" TEXT,
    "CUID"     INTEGER,
    "HORREF1"  TEXT,
    "HORREF1_" TEXT,
    "HORREF2"  TEXT,
    "HORREF2_" TEXT,
    "FGVXM"    INTEGER,
    "FGVXM_"   TEXT,
    "FINCBEFM" DOUBLE PRECISION,
    "FINC_EFM" TEXT,
    "FINCBEF1" INTEGER,
    "FINCBEF2" INTEGER,
    "FINCBEF3" INTEGER,
    "FINCBEF4" INTEGER,
    "FINCBEF5" INTEGER,
    "FINCBEFI" INTEGER,
    "FJSSDEDM" DOUBLE PRECISION,
    "FJSS_EDM" TEXT,
    "FJSSDED1" INTEGER,
    "FJSSDED2" INTEGER,
    "FJSSDED3" INTEGER,
    "FJSSDED4" INTEGER,
    "FJSSDED5" INTEGER,
    "FPVTXM"   INTEGER,
    "FPVTXM_"  TEXT,
    "FRRXM"    INTEGER,
    "FRRXM_"   TEXT,
    "FS_AMTXM" TEXT,
    "FS_A_TXM" TEXT,
    "FS_AMTX1" TEXT,
    "FS_AMTX2" TEXT,
    "FS_AMTX3" TEXT,
    "FS_AMTX4" TEXT,
    "FS_AMTX5" TEXT,
    "FS_AMTXI" INTEGER,
    "FSS_RRXM" DOUBLE PRECISION,
    "FSS__RXM" TEXT,
    "FSS_RRX1" INTEGER,
    "FSS_RRX2" INTEGER,
    "FSS_RRX3" INTEGER,
    "FSS_RRX4" INTEGER,
    "FSS_RRX5" INTEGER,
    "FSS_RRXI" INTEGER,
    "FSUPPXM"  DOUBLE PRECISION,
    "FSUPPXM_" TEXT,
    "FSUPPX1"  INTEGER,
    "FSUPPX2"  INTEGER,
    "FSUPPX3"  INTEGER,
    "FSUPPX4"  INTEGER,
    "FSUPPX5"  INTEGER,
    "FSUPPXI"  INTEGER,
    "FWAGEXM"  DOUBLE PRECISION,
    "FWAGEXM_" TEXT,
    "FWAGEX1"  INTEGER,
    "FWAGEX2"  INTEGER,
    "FWAGEX3"  INTEGER,
    "FWAGEX4"  INTEGER,
    "FWAGEX5"  INTEGER,
    "FWAGEXI"  INTEGER,
    "INC__NKM" TEXT,
    "JFS_AMTM" DOUBLE PRECISION,
    "JFS__MTM" TEXT,
    "JFS_AMT1" INTEGER,
    "JFS_AMT2" INTEGER,
    "JFS_AMT3" INTEGER,
    "JFS_AMT4" INTEGER,
    "JFS_AMT5" INTEGER,
    "OTHINXM"  TEXT,
    "OTHINXM_" TEXT,
    "OTHINX1"  TEXT,
    "OTHINX2"  TEXT,
    "OTHINX3"  TEXT,
    "OTHINX4"  TEXT,
    "OTHINX5"  TEXT,
    "OTHINXI"  INTEGER,
    "WELFRXM"  TEXT,
    "WELFRXM_" TEXT,
    "WELFRX1"  TEXT,
    "WELFRX2"  TEXT,
    "WELFRX3"  TEXT,
    "WELFRX4"  TEXT,
    "WELFRX5"  TEXT,
    "WELFRXI"  INTEGER,
    "PICKCODE" INTEGER,
    "LUMPB"    TEXT,
    "LUMPB_"   TEXT,
    "LUMPBX"   TEXT,
    "LUMPBX_"  TEXT,
    "OTHINB"   TEXT,
    "OTHINB_"  TEXT,
    "OTHINBX"  TEXT,
    "OTHINBX_" TEXT,
    "WELFRB"   TEXT,
    "WELFRB_"  TEXT,
    "WELFRBX"  TEXT,
    "WELFRBX_" TEXT,
    "PSU"      TEXT,
    "HIGH_EDU" DOUBLE PRECISION,
    "EITC"     TEXT,
    "EITC_"    TEXT,
    "FSMPFRMX" INTEGER,
    "FSMP_RMX" TEXT,
    "FSMPFRX1" INTEGER,
    "FSMPFRX2" INTEGER,
    "FSMPFRX3" INTEGER,
    "FSMPFRX4" INTEGER,
    "FSMPFRX5" INTEGER,
    "FSMPFRXI" INTEGER,
    "FSMPFRXM" INTEGER,
    "INTRDVB"  TEXT,
    "INTRDVB_" TEXT,
    "INTRDVBX" TEXT,
    "INTR_VBX" TEXT,
    "INTRDVX"  TEXT,
    "INTRDVX_" TEXT,
    "INTRDVX1" TEXT,
    "INTRDVX2" TEXT,
    "INTRDVX3" TEXT,
    "INTRDVX4" TEXT,
    "INTRDVX5" TEXT,
    "INTRDVXI" INTEGER,
    "INTRDVXM" TEXT,
    "NETRENTB" TEXT,
    "NETR_NTB" TEXT,
    "NETRENTX" TEXT,
    "NETR_NTX" TEXT,
    "NETRNTBX" TEXT,
    "NETR_TBX" TEXT,
    "NETRENT1" TEXT,
    "NETRENT2" TEXT,
    "NETRENT3" TEXT,
    "NETRENT4" TEXT,
    "NETRENT5" TEXT,
    "NETRENTI" INTEGER,
    "NETRENTM" TEXT,
    "OTHREGB"  TEXT,
    "OTHREGB_" TEXT,
    "OTHREGBX" TEXT,
    "OTHR_GBX" TEXT,
    "OTHREGX"  TEXT,
    "OTHREGX_" TEXT,
    "OTHREGX1" TEXT,
    "OTHREGX2" TEXT,
    "OTHREGX3" TEXT,
    "OTHREGX4" TEXT,
    "OTHREGX5" TEXT,
    "OTHREGXI" INTEGER,
    "OTHREGXM" TEXT,
    "RETSRVBX" TEXT,
    "RETS_VBX" TEXT,
    "RETSURVB" TEXT,
    "RETS_RVB" TEXT,
    "RETSURVX" TEXT,
    "RETS_RVX" TEXT,
    "RETSURV1" TEXT,
    "RETSURV2" TEXT,
    "RETSURV3" TEXT,
    "RETSURV4" TEXT,
    "RETSURV5" TEXT,
    "RETSURVI" INTEGER,
    "RETSURVM" TEXT,
    "ROYESTB"  TEXT,
    "ROYESTB_" TEXT,
    "ROYESTBX" TEXT,
    "ROYE_TBX" TEXT,
    "ROYESTX"  TEXT,
    "ROYESTX_" TEXT,
    "ROYESTX1" TEXT,
    "ROYESTX2" TEXT,
    "ROYESTX3" TEXT,
    "ROYESTX4" TEXT,
    "ROYESTX5" TEXT,
    "ROYESTXI" INTEGER,
    "ROYESTXM" TEXT,
    "FSMP_RXM" TEXT,
    "INTR_VXM" TEXT,
    "NETR_NTM" TEXT,
    "OTHR_GXM" TEXT,
    "RETS_RVM" TEXT,
    "ROYE_TXM" TEXT,
    "INT_HOME" TEXT,
    "INT_PHON" TEXT,
    "INT__OME" TEXT,
    "INT__HON" TEXT,
    "DIVISION" TEXT,
    "HISP_REF" INTEGER,
    "HISP2"    TEXT);
""")

database.read_csv("FMLD_RAW", fmld_fnames)

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

# The sniffer will interpret NEWID as
# as a DOUBLE PRECISION column. But we want it
# to be treated as TEXT.
# We can just copy and paste the original
# query suggested by the sniffer and
# make the modifications we want.
database.execute("""
DROP TABLE IF EXISTS "MEMD_RAW";

CREATE TABLE "MEMD_RAW"(
    "OCCULIST" TEXT,
    "HRSPERWK" TEXT,
    "WKS_WRKD" TEXT,
    "EMPLTYPE" TEXT,
    "MARITAL"  INTEGER,
    "HISPANIC" TEXT,
    "WHYNOWRK" TEXT,
    "MEMBRACE" INTEGER,
    "SEX"      INTEGER,
    "HRSP_RWK" TEXT,
    "WKS__RKD" TEXT,
    "EMPL_YPE" TEXT,
    "HISP_NIC" TEXT,
    "WHYN_WRK" TEXT,
    "OCCU_IST" TEXT,
    "NEWID"    TEXT,
    "AGE"      INTEGER,
    "AGE_"     TEXT,
    "ANGVX"    TEXT,
    "ANGVX_"   TEXT,
    "ANPVTX"   TEXT,
    "ANPVTX_"  TEXT,
    "ANRRX"    TEXT,
    "ANRRX_"   TEXT,
    "CU_CODE1" INTEGER,
    "EDUCA"    TEXT,
    "EDUCA_"   TEXT,
    "GROSPAYX" TEXT,
    "GROS_AYX" TEXT,
    "GVX"      TEXT,
    "GVX_"     TEXT,
    "IRAX"     TEXT,
    "IRAX_"    TEXT,
    "JSSDEDX"  TEXT,
    "JSSDEDX_" TEXT,
    "MEMBNO"   INTEGER,
    "PVTX"     TEXT,
    "PVTX_"    TEXT,
    "RRX"      TEXT,
    "RRX_"     TEXT,
    "SCHLNCHQ" TEXT,
    "SCHL_CHQ" TEXT,
    "SCHLNCHX" TEXT,
    "SCHL_CHX" TEXT,
    "SLFEMPSS" TEXT,
    "SLFE_PSS" TEXT,
    "SS_RRX"   TEXT,
    "SS_RRX_"  TEXT,
    "SUPPX"    TEXT,
    "SUPPX_"   TEXT,
    "US_SUPP"  TEXT,
    "US_SUPP_" TEXT,
    "WAGEX"    TEXT,
    "WAGEX_"   TEXT,
    "SS_RRQ"   TEXT,
    "SS_RRQ_"  TEXT,
    "SOCRRX"   TEXT,
    "SOCRRX_"  TEXT,
    "ARM_FORC" TEXT,
    "ARM__ORC" TEXT,
    "IN_COLL"  TEXT,
    "IN_COLL_" TEXT,
    "MEDICARE" TEXT,
    "MEDI_ARE" TEXT,
    "PAYPERD"  TEXT,
    "PAYPERD_" TEXT,
    "HORIGIN"  INTEGER,
    "RC_WHITE" TEXT,
    "RC_W_ITE" TEXT,
    "RC_BLACK" TEXT,
    "RC_B_ACK" TEXT,
    "RC_NATAM" TEXT,
    "RC_N_TAM" TEXT,
    "RC_ASIAN" TEXT,
    "RC_A_IAN" TEXT,
    "RC_PACIL" TEXT,
    "RC_P_CIL" TEXT,
    "RC_OTHER" TEXT,
    "RC_O_HER" TEXT,
    "RC_DK"    TEXT,
    "RC_DK_"   TEXT,
    "ANGVXM"   TEXT,
    "ANGVXM_"  TEXT,
    "ANPVTXM"  TEXT,
    "ANPVTXM_" TEXT,
    "ANRRXM"   TEXT,
    "ANRRXM_"  TEXT,
    "JSSDEDXM" TEXT,
    "JSSD_DXM" TEXT,
    "JSSDEDX1" TEXT,
    "JSSDEDX2" TEXT,
    "JSSDEDX3" TEXT,
    "JSSDEDX4" TEXT,
    "JSSDEDX5" TEXT,
    "SLFEMPSM" TEXT,
    "SLFE_PSM" TEXT,
    "SLFEMPS1" TEXT,
    "SLFEMPS2" TEXT,
    "SLFEMPS3" TEXT,
    "SLFEMPS4" TEXT,
    "SLFEMPS5" TEXT,
    "SOCRRXM"  TEXT,
    "SOCRRXM_" TEXT,
    "SOCRRX1"  TEXT,
    "SOCRRX2"  TEXT,
    "SOCRRX3"  TEXT,
    "SOCRRX4"  TEXT,
    "SOCRRX5"  TEXT,
    "SS_RRXM"  TEXT,
    "SS_RRXM_" TEXT,
    "SS_RRX1"  TEXT,
    "SS_RRX2"  TEXT,
    "SS_RRX3"  TEXT,
    "SS_RRX4"  TEXT,
    "SS_RRX5"  TEXT,
    "SS_RRXI"  TEXT,
    "SUPPXM"   TEXT,
    "SUPPXM_"  TEXT,
    "SUPPX1"   TEXT,
    "SUPPX2"   TEXT,
    "SUPPX3"   TEXT,
    "SUPPX4"   TEXT,
    "SUPPX5"   TEXT,
    "SUPPXI"   TEXT,
    "WAGEXM"   TEXT,
    "WAGEXM_"  TEXT,
    "WAGEX1"   TEXT,
    "WAGEX2"   TEXT,
    "WAGEX3"   TEXT,
    "WAGEX4"   TEXT,
    "WAGEX5"   TEXT,
    "WAGEXI"   TEXT,
    "SS_RRB"   TEXT,
    "SS_RRB_"  TEXT,
    "SS_RRBX"  TEXT,
    "SS_RRBX_" TEXT,
    "SUPPB"    TEXT,
    "SUPPB_"   TEXT,
    "SUPPBX"   TEXT,
    "SUPPBX_"  TEXT,
    "WAGEB"    TEXT,
    "WAGEB_"   TEXT,
    "WAGEBX"   TEXT,
    "WAGEBX_"  TEXT,
    "ASIAN"    TEXT,
    "ASIAN_"   TEXT,
    "OCCUEARN" TEXT,
    "PAYSTUB"  TEXT,
    "PAYSTUB_" TEXT,
    "SEMPFRM"  TEXT,
    "SEMPFRM_" TEXT,
    "SEMPFRMX" TEXT,
    "SEMP_RMX" TEXT,
    "SMPFRMB"  TEXT,
    "SMPFRMB_" TEXT,
    "SMPFRMBX" TEXT,
    "SMPF_MBX" TEXT,
    "SEMPFRM1" TEXT,
    "SEMPFRM2" TEXT,
    "SEMPFRM3" TEXT,
    "SEMPFRM4" TEXT,
    "SEMPFRM5" TEXT,
    "SEMPFRMI" TEXT,
    "SEMPFRMM" TEXT,
    "SEMP_RMM" TEXT,
    "SOCSRRET" TEXT,
    "SOCS_RET" TEXT,
    "WKSTATUS" TEXT);
""")

database.read_csv("MEMD_RAW", memd_fnames)

# -----------------------------------------------------------------------------
# Do the preprocessing - note that names in PostgreSQL in always a good idea to
# put names of tables and columns into quotation marks.

database.execute("""
    DROP TABLE IF EXISTS "EXPD";

    CREATE TABLE "EXPD" AS
    SELECT CASE WHEN "GIFT"=2 THEN 0 ELSE 1 END AS "TARGET",
           "EXPNYR" || '/' || "EXPNMO" || '/' || '01' AS "TIME_STAMP",
           "NEWID",
           "EXPNYR",
           CAST("EXPNMO" AS INT) AS "EXPNMO",
           "COST",
           substr("UCC", 1, 1) AS "UCC1",
           substr("UCC", 1, 2) AS "UCC2",
           substr("UCC", 1, 3) AS "UCC3",
           substr("UCC", 1, 4) AS "UCC4",
           substr("UCC", 1, 5) AS "UCC5",
           substr("UCC", 1, 6) AS "UCC"
    FROM "EXPD_RAW"
    WHERE "EXPNMO" != '';
""")

# -----------------------------------------------------------------------------
# Preprocess MEMD.

database.execute("""
    DROP TABLE IF EXISTS "MEMD";

    CREATE TABLE "MEMD" AS
    SELECT "MARITAL",
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
           "WKSTATUS",
           "AGE",
           "WAGEX",
           "NEWID",
           '2015/01/01' AS "TIME_STAMP"
    FROM "MEMD_RAW"
""")

# -----------------------------------------------------------------------------
# Make POPULATION TABLE

database.execute("""
    DROP TABLE IF EXISTS "POPULATION_ALL";

    CREATE TABLE "POPULATION_ALL" AS 
    SELECT t1.*,
           t2."INC_RANK",
           t2."INC_RNK1",
           t2."INC_RNK2",
           t2."INC_RNK3",
           t2."INC_RNK4",
           t2."INC_RNK5",
           t2."INC_RNKM"
    FROM "EXPD" t1
    LEFT JOIN "FMLD_RAW" t2
    ON t1."NEWID"= t2."NEWID";
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

