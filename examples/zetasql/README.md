# ZetaSQL Differential Privacy extension

This directory contains a command-line interface (CLI) and an example codelab
for performing (ε, δ)-differentially private (DP) data analyses using ZetaSQL.
This CLI provides direct access to the implementation of Wilson et al.'s PETS
2019 paper on [Differentially Private SQL with Bounded User Contribution](https://arxiv.org/pdf/1909.01917.pdf).

The remainder of this document will describe the usage of the ZetaSQL DP CLI. A
more comprehensive codelab illustrating both the usage of the CLI as well as
explaining some of the DP concepts and results can be found [here](codelab.mb).

## How to Build

In order to run the ZetaSQL DP CLI, you need to install Bazel version 1.0.0, if
you don't have it already. To install Bazel, follow the
[instructions for your platform on the Bazel website](https://docs.bazel.build/versions/master/install.html)
If this requires you to manage multiple Bazel versions, consider using
[Bazelisk](https://docs.bazel.build/versions/master/updating-bazel.html#managing-bazel-versions-with-bazelisk).

You also need to install Git, if you don't have it already.
[Follow the instructions for your platform on the Git website.](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).

Once you've installed Bazel and Git, open a Terminal and clone the differential
privacy directory into a local folder:
```git clone https://github.com/google/differential-privacy.git```

Navigate into the ```differential-privacy/examples/zetasql``` folder that was
just created, and build the ZetaSQL DP CLI and dependencies using Bazel:
``` bazel build :execute_query ```

Once built, the ZetaSQL DP CLI should be ready to use.

## Usage

The ZetaSQL DP CLI can be run from the
```differential-privacy/examples/zetasql``` with the following command:

```
bazel run :execute_query -- --data_set=<path_to_csv_file> --userid_col=<userid_column_name_in_data_set> <sql_statement>
```

The command requires the following arguments:

* ```data_set``` is the path to a file of comma-separated values (CSV) that
represents the data upon which the query should be executed. The first row is
interpreted as a header row containing the names of each of the columns. ZetaSQL
will input the CSV data into STRING columns in a table of the same name as the file
(without the ".csv" extension, if it exists). For example, the file
```data/day_data.csv``` will result in a table named ```day_data```.
* ```userid_col``` is the column in the ```data_set``` file that should be used
as the privacy unit, which is typically a column of identifiers for the users
that we want differential privacy to protect. For more information on what
privacy units could be, see the [codelab](codelab.md) and other examples in
this differential privacy library.
* The last argument should be the SQL query to execute on the ```data_set```.
This query must specify values for the DP parameters ```epsilon```, ```delta```,
and ```kappa```[^params] (see the
[ZetaSQL documentation](https://github.com/google/zetasql/anonymization_syntax.md#anon_kappa)
for more information). In queries that contain a GROUP BY clause, ```kappa```
is the maximum number of different groups (i.e., partitions) to which each user
may contribute data. See the [codelab](codelab.md) for additional information
and examples.

To illustrate the usage of the ZetaSQL DP CLI, below is an example command
querying the data in [data/day_data.csv](data/day_data.csv) of visitors to a
restaurant within a day, with the time they entered, how long they stayed, and
how many euros they spent. The following command queries the data for the DP
average of how long visitors stayed in the restaurant and the DP sum of how much
money people spent in the restaurant each hour:

```
bazel run :execute_query -- --data_set=data/day_data.csv --userid_col=VisitorId
'SELECT WITH ANONYMIZATION OPTIONS(epsilon=1, delta=1e-10, kappa=1)
 TIME_TRUNC(PARSE_TIME("%I:%M%p", `Time entered`), HOUR) AS `Hour entered`,
 ANON_AVG(CAST (`Time spent (mins)` AS INT32) CLAMPED BETWEEN 10 AND 120) AS `DP average time spent (mins)`,
 ANON_SUM(CAST (`Money spent (euros)` AS INT32) CLAMPED BETWEEN 10 AND 50) AS `DP total money spent (euros)` 
 FROM day_data GROUP BY `Hour entered`'
```

Note that, since all data input from ```data_set``` are stored in the SQL table
as STRING, it is necessary to CAST columns to the desired numerical data type
before performing the desired DP aggregation.

For more comprehensive explanations and examples of how to construct DP queries
using various anonymization aggregation functions with the ZetaSQL DP CLI, see
the [codelab](codelab.md).

[^params]: All three privacy parameters, epsilon, delta, and kappa, *must* be specified for the ZetaSQL DP CLI to provide (ε, δ)-differentially privacy, despite the ZetaSQL specification stating that some parameters are optional.
