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

import pathlib

import getml.engine as engine

# ----------------

engine.set_project("examples")

# ----------------
# Create a data frame from a JSON string

json_str = """{
    "names": ["patrick", "alex", "phil", "ulrike"],
    "column_01": [2.4, 3.0, 1.2, 1.4],
    "join_key": ["0", "1", "2", "3"],
    "time_stamp": ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"]
}"""

# ----------------

my_df = engine.DataFrame(
    "MY DF",
    categorical=["names"],
    join_keys=["join_key"],
    numerical=["column_01"],
    time_stamps=["time_stamp"]
)

my_df.from_json(
    json_str
)

# ----------------

col1 = my_df.numerical("column_01")

# ----------------

col2 = 2.0 - col1

my_df.add_numerical(col2, "column_02")

# ----------------

col3 = (col1 + 2.0*col2) / 3.0

my_df.add_numerical(col3, "column_03")

# ----------------

col4 = col1 ** -col3

my_df.add_numerical(col4, "column_04")

# ----------------

col5 = col1 ** 2.0

my_df.add_numerical(col5, "column_05")

# ----------------

col6 = 2.0 ** col1

my_df.add_numerical(col6, "column_06")

# ----------------

col7 = (col1 * 100000.0).sqrt()

my_df.add_numerical(col7, name="column_07", unit="time stamp, comparison only")

# ----------------

col8 = col1 % 0.5

my_df.add_numerical(col8, "column_08")

# ----------------

col9 = col1.erf()

my_df.add_numerical(col9, "column_09")


# ----------------

col10 = col1.ceil()

my_df.add_time_stamp(col10, "column_10")

# ----------------

col11 = col7.year()

my_df.add_discrete(col11, "year", "year, comparison only")

# ----------------

col12 = col7.month()

my_df.add_discrete(col12, "month")

# ----------------

col13 = col7.day()

my_df.add_discrete(col13, "day")

# ----------------

col14 = col7.hour()

my_df.add_discrete(col14, "hour")

# ----------------

col15 = col7.minute()

my_df.add_discrete(col15, "minute")

# ----------------

col16 = col7.second()

my_df.add_discrete(col16, "second")

# ----------------

col17 = col7.weekday()

my_df.add_discrete(col17, "weekday")

# ----------------

col18 = col7.yearday()

my_df.add_discrete(col18, "yearday")

# ----------------

col19 = my_df.categorical("names")

# ----------------

col20 = col19.substr(4, 3)

my_df.add_categorical(col20, "short_names")

# ----------------

col21 = "user-" + col19 + "-" + col20

my_df.add_categorical(col21, "new_names")

# ----------------

col22 = col17.to_str()

my_df.add_categorical(col22, "weekday")

# ----------------

col23 = my_df.time_stamp("time_stamp").to_str()

my_df.add_categorical(col23, "ts")

# ----------------

col24 = col19.contains("rick").to_str()

my_df.add_categorical(col24, "contains 'rick'?")

# ----------------

col25 = col19.update(col19.contains("rick"), "Patrick U")

col25 = col25.update(col19.contains("lex"), "Alex U")

my_df.add_categorical(col25, "update names")

# ----------------

col25 = my_df.rowid().to_str().to_ts(
    time_formats=["%Y-%m-%dT%H:%M:%s%z", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]
)

my_df.add_discrete(col25, "rowid")

# ----------------

my_df.rm_categorical("new_names")

# ----------------

home_folder = str(pathlib.Path.home()) + "/"

my_df.to_csv(home_folder + "MYDF.csv")

# ----------------
# You can write data frames to the data base - but be
# careful. Your data base might have stricter naming
# conventions for your columns than the data frames.

my_df.rm_categorical("contains 'rick'?")

my_df.rm_categorical("update names")

my_df.to_db("MYDF")

# ----------------

my_other_df = my_df.where("MY OTHER DF", (col1 > 1.3) | (col19 == "alex"))

my_other_df = my_df.where("MY OTHER DF", my_df.random(seed=100) > 0.5)
