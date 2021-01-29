# Example using differential privacy with ZetaSQL

## Restaurant

Imagine a fictional restaurant owner named Alice who would like to share
business statistics with her visitors. Alice knows when visitors enter the
restaurant and how much time and money they spend there. To ensure that
visitors' privacy is preserved, Alice decides to use this differential privacy
library.

## Count visits by hour of the day

In this example, Alice wants to share information with potential clients in
order to let them know when the restaurant is most busy.

For this, we will count how many visitors enter the restaurant at every hour of
a particular day. For simplicity, assume that a visitor comes to the restaurant
at most once a day. Thus, each visitor may only be present at most once in the
whole dataset, since the dataset represents a single day of restaurant visits.

The `data/day_data.csv` file contains visit data for a single day. It
includes the visitor’s ID, a timestamp of when the visitor entered the
restaurant, the duration of the visitor's visit to the restaurant (in minutes),
and the money the visitor spent at the restaurant.

Navigate to the `examples/zetasql` folder, build the code, and run it with the
following query.

Linux
```shell
$ cd examples/zetasql
$ bazel run execute_query -- --data_set=$(pwd)/data/day_data.csv --userid_col=VisitorId \
'SELECT WITH ANONYMIZATION OPTIONS(epsilon=1, delta=1e-5, kappa=1)
   TIME_TRUNC(PARSE_TIME("%I:%M%p", `Time entered`), HOUR) AS `Hour entered`,
   ANON_COUNT(* CLAMPED BETWEEN 0 AND 1) AS `Total Visitors (DP)`
 FROM day_data
 GROUP BY `Hour entered`'
```

This reads the daily statistics and calculates the number of visitors that
entered the restaurant every hour of the day in a differentially private way.

For illustration purposes, you can run the same query in a non-differentially
private way:

Linux
```shell
$ cd examples/zetasql
$ bazel run execute_query -- --data_set=$(pwd)/data/day_data.csv --userid_col=VisitorId \
'SELECT
   TIME_TRUNC(PARSE_TIME("%I:%M%p", `Time entered`), HOUR) AS `Hour entered`,
   COUNT(*) AS `Total Visitors (Raw)`
 FROM day_data
 GROUP BY `Hour entered`'
```

The image below illustrates the results. The blue (left) bars represent the
counts without anonymization while red (right) bars correspond to the private
(or *anonymized*) counts. Notice that the private counts slightly differ
from the non-private counts, though the overall trend is preserved: the
restaurant is more busy during lunch and dinner time.

![Daily counts](img/visits_per_hour.png)

Note that differential privacy involves adding *random noise* to the actual
data and hence, your own results will most likely be slightly different.

## Partitions and contributions

We say that the resulting aggregated data is split into *partitions*. The bar
chart for the private and non-private counts each have 12 partitions: one for
each hour the restaurant was visited.

More generally, a single partition represents a subset of aggregated data
corresponding to a given value of the aggregation criterion. Graphically, a
single partition is represented as a bar on the aggregated bar chart.

We say that a visitor *contributes* to a given partition if their data matches
the partition criterion. For example, if a visitor enters between 8 AM and 9 AM,
they contribute to the 8 AM partition.

Recall that in the example above, a visitor can enter the restaurant only
once per day. This implies the following *contribution bounds*.

*    **Kappa** is the limit of how many separate partitions to which a visitor
can contribute. In our example, a visitor can contribute to, at most, only
one partition. In other words, there is at most one time-slot when a visitor
with a given ID can enter the restaurant.
*   **Clamped between** sets the bounds on the input values that users (i.e.,
visitors) contribute to the DP aggregation. In the example above using
`ANON_COUNT`, a visitor can contribute either zero (i.e., a visitor may have not
entered the restaurant at that time) or at most once, meaning they may enter the
restaurant only once at a given hour.

Why is this important? Differential privacy adjusts the amount of noise to mask
the contributions of each visitor. The more contributions each user is allowed
to contribute to the data, the more noise is needed to protect users' privacy.

Next, we will demonstrate how to use the library in scenarios where:

*   visitors can contribute to multiple partitions;
*   contributed values can be greater than *1*; and
*   visitors can contribute to a partition multiple times.

## Count the visits per day of the week

The previous example made some over-simplifying assumptions. Now, let’s have a
look at the use-case where visitors can contribute to multiple partitions.

Imagine Alice decides to let visitors know what the busiest *days* at her
restaurant are. For this, she calculates how many people visit the restaurant
every day of the week. For simplicity, let’s assume a visitor enters the
restaurant at most once a day but may enter multiple days in a single week.

The file `data/week_data.csv` contains visit data for a week.
It includes the visitor’s ID, the visit duration (in minutes),
the money spent at the restaurant, and the day of the visit.

Navigate to the `examples/zetasql` folder, build the code, and run it with the
following query.

Linux
```shell
$ cd examples/zetasql
$ bazel run execute_query -- --data_set=$(pwd)/data/week_data.csv --userid_col=VisitorId \
'SELECT WITH ANONYMIZATION OPTIONS(epsilon = 1, delta = 1e-5, kappa = 3)
   Day,
   ANON_COUNT(* CLAMPED BETWEEN 0 AND 1) AS `Total Visitors (DP)`
 FROM week_data
 GROUP BY Day'
```

This calculates the number of visitors that entered the restaurant for each day
of the week in a differentially private way.

For illustration purposes, you can run the same query in a non-differentially
private way:

Linux
```shell
$ cd examples/zetasql
$ bazel run execute_query -- --data_set=$(pwd)/data/week_data.csv --userid_col=VisitorId \
'SELECT
   `Day`,
   COUNT(*) AS `Total Visitors (DP)`
 FROM week_data
 GROUP BY `Day`'
```

The results are illustrated in the image below.

![Counts per week day](img/visits_per_day.png)

Notice that the private values slightly differ from the actual ones, though the
overall trend is preserved.

Now, let’s take a closer look at the technical details. Speaking in terms of
*partitions* and *contributions*, the resulting bar chart has 7 partitions; one
for each day of the week. A visitor may enter the restaurant once a day and
hence contribute to a partition at most once. A visitor may enter the restaurant
several times a week and hence contribute to up to 7 partitions.

### Bounding the number of contributed partitions

The parameter `kappa` defines the maximum number of partitions to which a
visitor may contribute. One may notice that the value of `kappa` in our example
is 3 instead of 7.

Why is that? Differential privacy adds some amount of random noise to hide
the contributions of an individual. The more contributions each individual may
have, the larger the noise must be, to protect each individual's privacy. However,
this affects the utility of the data. In order to preserve the data's utility,
we made an approximate estimate of how many times a week, at most, a person may
visit a restaurant on average, and assumed that the value is around 3 instead of
scaling the noise by the factor of 7 (which would have required adding more
noise).

ZetaSQL processes the input data and discards all exceeding contributions (visits)
automatically, so no action is required from the user.

## Sum the revenue per day of the week

The previous example demonstrates how the contributed partitions are bounded.
Now, we will demonstrate how individual contributions are clamped. Imagine Alice
decides to calculate the sum of the restaurant's revenue per weekday in a
differentially private way. For this, she needs to sum the visitors’ daily
spending at the restaurant. For simplicity, let’s assume a visitor enters the
restaurant at most once a day but may enter multiple times a week (on different
days).

The `data/week_data.csv` file contains visit data for a week.
It includes the visitor’s ID, the visit duration (in minutes),
the money spent at the restaurant, and the day of the visit.

Navigate to the `examples/zetasql` folder, build the code and run it with the
following query.

Linux
```shell
$ cd examples/zetasql
$ bazel run execute_query -- --data_set=$(pwd)/data/week_data.csv --userid_col=VisitorId \
'SELECT WITH ANONYMIZATION OPTIONS(epsilon=1, delta=1e-5, kappa=3)
   `Day`,
   ANON_SUM(CAST (`Money spent (euros)` AS INT32) CLAMPED BETWEEN 10 AND 50) AS `Total money spent (DP)`
 FROM week_data
 GROUP BY `Day`'
```

This sums the amount of money visitors spend at the restaurant on each day of
the week in a differentially private way.

For illustration purposes, you can run the same query in a non-differentially
private way:

Linux
```shell
$ cd examples/zetasql
$ bazel run execute_query -- --data_set=$(pwd)/data/week_data.csv --userid_col=VisitorId \
'SELECT
   `Day`,
   SUM(CAST (`Money spent (euros)` AS INT32)) AS `Total money spent (Raw)`
 FROM week_data
 GROUP BY `Day`'
```

The results are illustrated in the image below.

![Daily sums](img/sums_per_day.png)

### Clamping individual contributions

The usage of `kappa` for `ANON_COUNT` is similar to its usage for `ANON_SUM`,
which is explained in the previous example. This section focuses on the *lower*
and *upper* bounds, expressed as `CLAMPED BETWEEN lower AND upper`. The `lower`
and `upper` bounds of `CLAMPED BETWEEN` define the bounds on the sum of each
users' per partition data contributions such that the sum of each user's
contributions will be automatically clamped to the specified bounds. Note that
the clamping is not applied on users' individual contributions per user (i.e.,
the amount of money a visitor spends on each visit) but on the sum of all such
contributions per user per partition (i.e., total money spent by a visitor each
day).

In other words, any sum that is below `lower` will be overridden to become the
same as `lower`, and any sum greater than `upper` will be reduced to be the
same as `upper`, so that the sum of all contributions for each visitor will be
within the bounds of `lower` and `upper`. This is needed for calculating the
sensitivity of the aggregation, and to scale the noise that will be added to the
sum accordingly.

### Choosing bounds

The lower and upper bounds affect the utility of the sum in two potentially
opposing ways: reducing the added noise, and preserving the utility. On the one
hand, the added noise is proportional to the maximum of absolute values of the
bounds. Thus, the closer the bounds are to zero, the less noise is added. On the
other hand, setting the lower and upper bound close to zero may mean that the
input values are clamped more aggressively, which can decrease utility as well.

### Automatic Bounds Approximation (Approx Bounds)

In case `CLAMPED BETWEEN` (i.e., `lower` and `upper` bounds on input values) is
not provided to `ANON_` aggregations, ZetaSQL will infer the bounds
automatically in a differentially private way from the data. See
[this](https://arxiv.org/abs/1909.01917) paper or [the C++ code](https://github.com/google/differential-privacy/blob/main/cc/algorithms/approx-bounds.h)
for details.

Note that ZetaSQL uses a portion of the privacy budget for automatic bounds
determination, meaning that it would need to add more noise to the aggregation
output to compensate for this.

## Count visits by certain duration

We will now demonstrate that just adding noise to raw data might be not enough
to preserve privacy, and that eliminating some partitions completely may be
necessary. This is referred to as *thresholding* or *partition selection*.

Imagine Alice wants to know how much time visitors typically spend at the
restaurant. Durations of visits can vary greatly. A visitor may be at the
restaurant for 31 minutes, while another may stay for a couple of hours. Instead
of considering the exact durations, Alice wants to look at the approximate
durations, where each visit duration is rounded up to the nearest multiple
of 10 minutes. Assume that a visitor may enter the restaurant at most once a
day, multiple times a week.

The `data/outliers_week_data.csv` file contains visit data for a week.
It includes the visitor’s ID, the visit duration (in minutes),
the money spent at the restaurant, and the day of the visit.

Navigate to the `examples/zetasql` folder, build the code and run it with the
following query.

Linux
```shell
$ cd examples/zetasql
$ bazel run execute_query -- --data_set=$(pwd)/data/outliers_week_data.csv --userid_col=VisitorId \
'SELECT WITH ANONYMIZATION OPTIONS(epsilon = 1, delta = 1e-5, kappa = 1)
   CAST(SAFE_MULTIPLY(CEIL(SAFE_DIVIDE(CAST(`Time spent (minutes)` AS DOUBLE), 10.0)), 10.0) AS INT32) AS `Time Spent`,
   ANON_COUNT(CAST(SAFE_MULTIPLY(CEIL(SAFE_DIVIDE(CAST(`Time spent (minutes)` AS DOUBLE), 10.0)), 10.0) AS INT32) CLAMPED BETWEEN 0 AND 3) AS `Total visitors (DP)`
 FROM outliers_week_data
 GROUP BY `Time Spent`'
```

This calculates the number of visitors who spent a certain amount of time (per
visit) during the week in a differentially private way.

For illustration purposes, you can run the same query in a non-differentially
private way:

Linux
```shell
$ cd examples/zetasql
$ bazel run execute_query -- --data_set=$(pwd)/data/outliers_week_data.csv --userid_col=VisitorId \
'SELECT
   CAST(SAFE_MULTIPLY(CEIL(SAFE_DIVIDE(CAST (`Time spent (minutes)` AS DOUBLE), 10.0)), 10.0) AS INT32) AS `Time Spent`,
   COUNT(CAST(SAFE_MULTIPLY(CEIL(SAFE_DIVIDE(CAST (`Time spent (minutes)` AS DOUBLE), 10.0)), 10.0) AS INT32)) AS `Total visitors (Raw)`
 FROM outliers_week_data
 GROUP BY `Time Spent`'
```

The results are illustrated in the image below.

![Duration counts](img/visits_per_duration.png)

Similar to the previous examples, private counts slightly differ from the
non-private counts. In addition, some partitions (e.g., 10 minutes and 180
minutes) do not appear in the private statistics. This is because some
partitions are dropped due to thresholding.

Let’s take a closer look at the technical details. Speaking in terms of
*partitions* and *contributions*, the resulting bar-plot for raw counts has 13
partitions and resulting bar-plot for anonymized counts has 11 partitions. Each
partition is a visit duration (i.e., the initial set of partitions consists of
all visit durations which occurred during the week, rounded up to 10 minutes). A
visitor may enter the restaurant at most once a day, multiple times a week. A
visitor may spend approximately the same time in each visit (e.g., a
fixed-duration lunch break), and hence all their visits over the week may
contribute to the same partition.

### Partition selection

Having too few privacy units (i.e., visitors) contributing to a partition, or
having too few results in a partition, can put users' privacy at risk, even after
adding noise to the data. More precisely, if the set of partitions is not known
in advance, the presence of a particular partition in the output can give an
attacker information about the users in the dataset. In order to protect against
such cases, we need to remove these partitions with too few contributions from
the output.

There are 2 approaches for doing this:

*   **Pre-aggregation partition selection**. Drop partitions which do not have
    a sufficient number of contributing privacy units. Note that these
    partitions may still be included in other aggregation functions' results,
    if there are sufficiently many contributing privacy units for that
    aggregation.
*   **Post-aggregation partition selection**. Apply a threshold to the result
    of an aggregation. This can be only used with count and sum aggregations.
    This threshold is independent from the data; it is calculated using given
    DP parameters. After calculating the statistic (either count or sum) for the
    partition, if the statistic value is less than the threshold, then the
    partition will be excluded from the final result.

ZetaSQL only supports pre-aggregation partition selection.

Note that ZetaSQL takes care of partition selection automatically when you do
an `ANON_` aggregation, so no action is needed on your side.
