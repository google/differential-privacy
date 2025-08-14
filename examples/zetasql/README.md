# ZetaSQL Differential Privacy extension

This directory contains a command-line interface (CLI) and an example codelab
for performing (ε, δ)-differentially private (DP) data analyses using ZetaSQL.
This CLI provides direct access to the implementation of Wilson et al.'s PETS
2019 paper on [Differentially Private SQL with Bounded User Contribution](https://arxiv.org/abs/1909.01917).

The ZetaSQL DP CLI currently only supports Linux and Mac OS on x86_64
architectures.

The remainder of this document will describe the usage of the ZetaSQL DP CLI. A
more comprehensive codelab illustrating both the usage of the CLI as well as
explaining some of the DP concepts and results can be found [here](codelab.md).

## How to Build

In order to run the ZetaSQL DP CLI, you need to install Bazel. We recommend
using
[Bazelisk](https://docs.bazel.build/versions/main/updating-bazel.html#managing-bazel-versions-with-bazelisk)
to manage different versions of Bazel.

Once you've installed Bazelisk and Git, open a Terminal and clone the differential
privacy directory into a local folder:
```sh
git clone https://github.com/google/differential-privacy.git
```

Alternatively, you could download the source code via:
```shell
https://github.com/google/differential-privacy/archive/main.zip
```

Navigate into the `differential-privacy/examples/zetasql` folder that was
just created, and build the ZetaSQL DP CLI and dependencies using Bazel:
```shell
bazelisk build //:execute_query
```

Once built, the ZetaSQL DP CLI should be ready to use.

## Usage

The ZetaSQL DP CLI can be run from the
`differential-privacy/examples/zetasql` folder with the following command:

```shell
bazelisk run //:execute_query -- <sql_statement>
```

The last argument should be the SQL statement to execute on the pre-defined
sample data stored in CSV files under the [./data](data/) folder.  See the
[ZetaSQL
documentation](https://github.com/google/zetasql/blob/master/docs/differential-privacy.md)
for more information on the query syntax and the [codelab](codelab.md) for
examples on querying the sample data.

To illustrate the usage of the ZetaSQL DP CLI, below is an example command
querying the data in [data/day_data.csv](data/day_data.csv) of visitors to a
restaurant within a day, with the time they entered, how long they stayed, and
how many euros they spent. The following command queries the data for the DP
average of how long visitors stayed in the restaurant and the DP sum of how much
money people spent in the restaurant each hour:

```shell
bazelisk run //:execute_query -- '
SELECT WITH DIFFERENTIAL_PRIVACY OPTIONS(
    epsilon=1, delta=1e-10, max_groups_contributed=1, privacy_unit_column=VisitorId)
  TIME_TRUNC(PARSE_TIME("%I:%M%p", `Time entered`), HOUR) AS `Hour entered`,
  AVG(CAST (`Time spent (mins)` AS INT32), contribution_bounds_per_group => (10, 120)) AS `DP average time spent (mins)`,
  SUM(CAST (`Money spent (euros)` AS INT32), contribution_bounds_per_group => (10, 50)) AS `DP total money spent (euros)`
FROM day_data
GROUP BY `Hour entered`'
```

Note that, since all data input from csv files are stored in the SQL table as
STRING, it is necessary to CAST columns to the desired numerical data type
before performing the desired DP aggregation.

For more comprehensive explanations and examples of how to construct DP queries
using various anonymization aggregation functions with the ZetaSQL DP CLI, see
the [codelab](codelab.md).

## Known Issues

1. Windows is currently not a supported configuration.
1. In case we are running out of memory, try using the
   `--evaluator_max_value_byte_size` and
   `--evaluator_max_intermediate_byte_size` flags to increase the max memory.

We will continue to publish updates and improvements to the library. Please
[reach out](https://github.com/google/differential-privacy#reach-out) to us with
any feedback.
