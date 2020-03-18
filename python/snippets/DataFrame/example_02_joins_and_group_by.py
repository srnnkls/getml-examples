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

import getml.data as data
import getml.engine as engine

# ----------------

engine.set_project("examples")

# ----------------

json_str1 = """{
    "names": ["patrick", "alex", "phil", "ulrike"],
    "column_01": [2.4, 3.0, 1.2, 1.4],
    "join_key": ["0", "1", "2", "3"],
    "time_stamp": ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04"]
}"""

# ----------------

json_str2 = """{
    "names": ["patrick", "alex", "phil", "johannes", "ulrike", "adil"],
    "column_01": [2.4, 3.0, 1.2, 1.4, 3.4, 2.2],
    "join_key": ["0", "1", "2", "2", "3", "4"],
    "time_stamp": ["2019-01-01", "2019-01-02", "2019-01-03", "2019-01-04", "2019-01-05", "2019-01-06"]
}"""

# ----------------

my_df1 = data.DataFrame(
    "MY DF 1",
    roles={
        "categorical": ["names"],
        "join_key": ["join_key"],
        "numerical": ["column_01"],
        "time_stamp": ["time_stamp"]}
).read_json(
    json_str1
)

# ----------------

my_df2 = data.DataFrame(
    "MY DF 2",
    roles={
        "categorical": ["names"],
        "join_key": ["join_key"],
        "numerical": ["column_01"],
        "time_stamp": ["time_stamp"]}
).read_json(
    json_str2
)

# ----------------

joined_df1 = my_df1.join(
    name="JOINED DF1",
    other=my_df2,
    join_key="join_key",
    cols=[
        my_df1["column_01"],
        my_df1["join_key"],
        my_df1["time_stamp"].alias("time_stamp1")
    ],
    other_cols=[
        my_df2["column_01"].alias("column_02"),
        my_df2["time_stamp"].alias("time_stamp2")
    ],
    how="left"
)

# ----------------
# You can do a subselection like this...

joined_df2 = my_df1.join(
    name="JOINED DF2",
    other=my_df2,
    join_key="join_key",
    cols=[
        my_df1["column_01"],
        my_df1["join_key"],
        my_df1["time_stamp"].alias("time_stamp1")
    ],
    other_cols=[
        my_df2["column_01"].alias("column_02"),
        my_df2["time_stamp"].alias("time_stamp2")
    ],
    how="left"
)

col1 = joined_df2["column_01"]
col2 = joined_df2["column_02"]

joined_df2 = joined_df2.where(
  "JOINED DF2",
  (col1 == col2)
)

# ----------------
# ...but this is more memory-efficient.

col1 = my_df1["column_01"]
col2 = my_df2["column_01"]

joined_df3 = my_df1.join(
    name="JOINED DF3",
    other=my_df2,
    join_key="join_key",
    cols=[
        my_df1["column_01"],
        my_df1["join_key"],
        my_df1["time_stamp"].alias("time_stamp1")
    ],
    other_cols=[
        my_df2["column_01"].alias("column_02"),
        my_df2["time_stamp"].alias("time_stamp2")
    ],
    how="left",
    where=(col1 == col2)
)

# ----------------

json_str = """{
    "names": ["patrick", "alex", "phil", "ulrike", "patrick", "alex", "phil", "ulrike", "NULL"],
    "column_01": [2.4, 3.0, 1.2, 1.4, 3.4, 2.2, 10.2, 13.5, 11.0],
    "join_key": ["0", "1", "2", "2", "3", "3", "4", "4", "4"]
}"""

# ----------------

my_df3 = data.DataFrame(
    "MY DF3",
    roles={
        "categorical": ["names"],
        "join_key": ["join_key"],
        "numerical": ["column_01"]}
).read_json(
    json_str
)

# ----------------

col1 = my_df3["column_01"]

names = my_df3["names"]

# Note that NULL is never aggregated
grouped_df = my_df3.group_by(
    "join_key",
    "GROUPED DF",
    [col1.avg(alias="column_01_avg"),
     col1.count(alias="column_01_count"),
     (col1 + 3.0).max(alias="column_01_plus_3_max"),
     col1.median(alias="column_01_median"),
     col1.min(alias="column_01_min"),
     col1.stddev(alias="column_01_stddev"),
     col1.sum(alias="column_01_sum"),
     col1.var(alias="column_01_var"),
     names.count(alias="names_count"),
     names.count_distinct(alias="names_count_distinct")]
)

# ----------------

my_df3.to_db("MYDF3")

grouped_df.to_db("GROUPEDDF")

# ----------------

col1 = my_df1["column_01"]

count_greater_than_two = (col1 > 2.0).as_num().sum().get()

print(count_greater_than_two)

# ----------------

engine.delete_project("examples")
