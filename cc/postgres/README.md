# Anonymous Functions PostgreSQL Extension

This subdirectory contains a PostgreSQL extension providing several epsilon-DP
aggregate functions. We will refer to them as the anonymous functions.

## Setup

* Install Postgres 11 using the source code.
  *  Source: https://www.postgresql.org/ftp/source/
  *  Instructions: https://www.postgresql.org/docs/9.3/install-short.html

* From the `cc` directory of the differential library repo, run

    ```
    ./postgres/install_extension.sh
    ```

  *  Note that this script invokes `sudo install` to copy the compiled extension
     into the postgres system folders.

* Start the Postgres server.

* In PostgreSQL, load the extension by calling

    ```
    CREATE EXTENSION anon_func;
    ```

### Common Issues

There are several known setup problems; we list suggested solutions for them
below. This library was built for Linux and MacOS; there may be errors using
Windows.


##### pg_config: command not found

While executing `install_extension.sh`, you may see the following error:

```
postgres/install_extension.sh: line 26: pg_config: command not found
```

This indicates that Postgres is either incorrectly installed or the pg_config
utility is not found in your system path. Verify that you've installed Postgres
according to the installation instructions listed and that pg_config is included
in your system path and try again.


##### Extension files missing

While loading the extension, you might see an error like the following:

```
could not access file "anon_func": No such file or directory
```
or

```
could not open extension control file "/usr/local/share/postgresql/extension/anon_func.control": No such file or directory
```

The installation script assumes you installed Postgres using the source code and
the instructions listed above. However, if you are using a different
installation, the paths to the lib and extension directories may be different.
If so, move the files to the proper locations. For the files
`anon_func.control` and `anon_func--1.0.0.sql`, move them to the proper
extension directory, e.g.

```
mv $PG_DIR/share/extension/anon_func.control /usr/local/share/postgresql/extension/
```

For the file `anon_func.so`, move it to the lib directory, which can often be
found by running `pg_config --pkglibdir`. However, if you have multiple
installations or have uninstalled previous versions, the path may be
erroneous.

```
mv $PG_DIR/lib/anon_func.so `pg_config  --pkglibdir`
```


## Anonymous Functions

We offer a suite of anonymous functions. Each function wraps around the
corresponding algorithm in the differential privacy library. Notice that these
functions, like their algorithms, assume that each entry is owned by a distinct
user. In addition, due to the limitations of defining aggregates, the functions
return NULL for the empty set. This means the functions are not DP for the empty
set.

The first argument for each function is the column over which the aggregation is
performed. Each function may also take a couple of additional parameters. They
must be passed as literal values.

  *  `epsilon`: The differential privacy parameter for the function.
  *  `lower`: A lower bound for the input data set.
  *  `upper`: An upper bound for the input data set.

The lower and upper bounds are used to determine the sensitivity for the
anonymous function. In the event that manual bounds are not entered, some
functions can automatically infer the bounds, provided that there is enough
input data. If there is not enough data, an error message to that effect will
be displayed. For more information on automatic bounding, see the [ApproxBounds
documentation](https://github.com/google/differential-privacy/blob/main/cc/docs/algorithms/approx-bounds.md)

### Count

```
ANON_COUNT(column)
ANON_COUNT(column, epsilon)
```

The count function can be called on a `column` of any type. The epsilon can be
optionally provided; otherwise, the default epsilon is used. An integer count is
returned.

### Sum, Average, Variance, Standard Deviation

```
ANON_SUM(column)
ANON_SUM(column, epsilon)
ANON_SUM_WITH_BOUNDS(column, lower, upper)
ANON_SUM_WITH_BOUNDS(column, lower, upper, epsilon)
```

`ANON_SUM` can be called with any numeric type. For integer types, the sum
returned is an integer. For floating point types, it is a double. If `ANON_SUM`
is called without bounds, then the data is automatically bounded. If
`ANON_SUM_WITH_BOUNDS` is called, then the data is bounded using the provided
explicit bounds. Like count, epsilon is optionally configurable.

```
ANON_AVG(column)
ANON_AVG(column, epsilon)
ANON_AVG_WITH_BOUNDS(column, lower, upper)
ANON_AVG_WITH_BOUNDS(column, lower, upper, epsilon)

ANON_VAR(column)
ANON_VAR(column, epsilon)
ANON_VAR_WITH_BOUNDS(column, lower, upper)
ANON_VAR_WITH_BOUNDS(column, lower, upper, epsilon)

ANON_STDDEV(column)
ANON_STDDEV(column, epsilon)
ANON_STDDEV_WITH_BOUNDS(column, lower, upper)
ANON_STDDEV_WITH_BOUNDS(column, lower, upper, epsilon)
```

The `ANON_AVG`, `ANON_VAR`, and `ANON_STDDEV` functions are like `ANON_SUM`, but
the return type is always double.

### Ntile

```
ANON_NTILE(column, percentile, lower, upper)
ANON_NTILE(column, percentile, lower, upper, epsilon)
```

`ANON_NTILE` accepts any numeric type. For integer types, an integer will be
returned. Otherwise, a double is returned. Unlike the other bounded functions,
`ANON_NTILE` requires bounds. Automatic bounding is not supported.


## User-Level Differentially Private Queries

To write user-level differentially private queries, a two-stage aggregation may
be used, as described in "Differentially Private SQL with Bounded User
Contribution" (Wilson et al.). In this section, we provide several examples for
manually rewriting regular PostgreSQL queries into queries with user-level
differential privacy. For an in-depth description of the transformations made,
refer to the Wilson et al. paper.


### Simple Count

Imagine a table that records each instance of a person eating a fruit, call it
`FruitEaten`.

Column   | Type        | Description                           |
-------- | ----------- | ------------------------------------- |
uid      | integer     | Uniquely identifies a person.         |
fruit    | varchar(20) | The name of the fruit the person ate. |

Create the table and import the data from the csv file we have provided in this
directory called `fruiteaten.csv`. Make sure to change the file path below to
point to where you cloned the directory.

```
CREATE TABLE FruitEaten (
  uid integer,
  fruit character varying(20)
);
COPY fruiteaten(uid, fruit) FROM 'fruiteaten.csv' DELIMITER ',' CSV HEADER;
```

In this table, each row represents one fruit eaten. So if person `1` eats two
`apple`s, then there will be two rows in the table with column values
`(1, apple)`. Consider a simple query counting how many of each fruit have been
eaten.

```
SELECT fruit, COUNT(fruit)
FROM FruitEaten
GROUP BY fruit;
```

Suppose that instead of getting the regular count, we want the differentially
private count with the privacy parameter ε=ln(3). The final product of the query
rewrite would be

```
SELECT result.fruit, result.number_eaten
FROM (
  SELECT per_person.fruit,
    ANON_SUM(per_person.fruit_count, LN(3)/2) as number_eaten,
    ANON_COUNT(uid, LN(3)/2) as number_eaters
    FROM(
      SELECT * , ROW_NUMBER() OVER (
        PARTITION BY uid
        ORDER BY random()
      ) as row_num
      FROM (
        SELECT fruit, uid, COUNT(fruit) as fruit_count
        FROM FruitEaten
        GROUP BY fruit, uid
      ) as per_person_raw
    ) as per_person
  WHERE per_person.row_num <= 5
  GROUP BY per_person.fruit
) as result
WHERE result.number_eaters > 50;

```
As we can see, there are four `SELECTS` in the query. We will explain them from
inner-most to outer-most. The following steps were taken to rewrite the query:

  * Construct the 1st and inner-most `SELECT`, aliased as `per_person_raw`.

    ```
    SELECT fruit, uid, COUNT(fruit) as fruit_count
    FROM FruitEaten
    GROUP BY fruit, uid;
    ```

    For each person, count how many of each fruit that person ate.

  * Construct the 2nd `SELECT`, aliased as `per_person`.

    ```
    SELECT *, ROW_NUMBER() OVER (
      PARTITION BY uid
      ORDER BY random()
    ) as row_num
    FROM per_person_raw;
    ```

    For each person, `per_person_raw` contains rows corresponding to the fruits
    they have eaten. We shuffle these rows and assign them a row number. This
    will allow us to effectively reservior sample rows for each user by
    filtering by row number in the next step. This is similar to C_u
    thresholding in Wilson et al.

  * Construct the 3rd `SELECT`, aliased as `result`.

    ```
    SELECT per_person.fruit,
           ANON_SUM(per_person.fruit_count, LN(3)/2) as number_eaten,
           ANON_COUNT(uid, LN(3)/2) as number_eaters
    FROM per_person
    WHERE per_person.row_num <= 5
    GROUP BY per_person.fruit;
    ```

    First, for each person, we ensure that the person only contributed to 5
    fruit groups by filtering on the randomized row number generated in the
    previous step. Then, for each fruit, we sum the number of the fruit that
    each person ate. We also count the number of people who ate that fruit. This
    will allow us to ensure we only release the sums which enough people
    contributed to.

  * In order to anonymously grab the sum and count, we replace the aggregates
    `SUM` and `COUNT` with `ANON_SUM` and `ANON_COUNT`, respectively. Note that
    since we are performing two anonymous aggregations, we split our privacy
    parameter between them, using ε=ln(3) for both.

  * Construct the 4th and outer-most `SELECT`.

    ```
    SELECT result.fruit, result.number_eaten
    FROM result
    WHERE result.number_eaters > 50;

    ```

    For any fruit where the number of eaters was less than 50, discard the
    output result. This is similar to τ-thresholding in Wilson et al. In
    addition, we drop the `number_eaters` count, so that it does not display in
    the output. Dropping the count of unique users is not neccesary for
    differential privacy.


#### Multiple Aggregations

In our simple count example, we used a query containing a single anonymous
function. For a query with N anonymous function calls, and with a desired
total privacy parameter of ε, we need to use ε/(N+1) as the privacy parameter
for each aggregation. This is for the N requested calls plus the additional
anonymous unique-user count. For instance, consider the following dummy query:

```
SELECT COUNT(col1), SUM(col2)
FROM Table;
```

Suppose we want to use the privacy paramter ε=M. In the rewritten query, we have
to make the following replacements

Original       | Replacement                      |
-------------- | -------------------------------- |
`COUNT(col1)`  | `ANON_COUNT(col1, M/3)`          |
`SUM(col2)`    | `ANON_SUM(col2, M/3)`            |

This is because we have two requested anonymous functions, and an additional
anonymous unique user count required when we perform the rewrite.


#### Bounding User Contribution

Consider again our fruit-eating example. Suppose we want to restrict the
contribution of each person to the fruit-eaten counts by `5`. So if a person has
eaten more than `5` fruit, we want to count it as that they have eaten `5`
fruit. To do this, add lower and upper bounds on the anonymous functions:

Original                                     | Replacement                                       |
-------------------------------------------- | ------------------------------------------------- |
`ANON_SUM(per_person.fruit_count, LN(3)/2)`  | `ANON_SUM(per_person.fruit_count, 0, 5, LN(3)/2)` |


### Query With Joins

In this section we will add a join to our query. In addition to the `FruitEaten`
table, consider the following table, which we will call `Shirts`.

Column   | Type        | Description                           |
-------- | ----------- | ------------------------------------- |
uid      | integer     | Uniquely dentifies a person.          |
color    | varchar(20) | The name of the person's shirt color. |

Create the table and import the data provided by `shirts.csv`.  Make sure to
change the file path below to point to where you cloned the directory.

```
CREATE TABLE Shirts (
  uid integer,
  color character varying(20)
);
COPY shirts(uid, color) FROM 'shirts.csv' DELIMITER ',' CSV HEADER;
```

Let's say we want to find out, for each shirt color, how many fruit all the
people wearing that shirt color ate, altogether.

```
SELECT color, COUNT(fruit)
FROM FruitEaten f INNER JOIN Shirts s ON (f.uid = s.uid)
GROUP BY color;
```

We rewrite the query into its differentially private version:

```
SELECT result.color, result.number_eaten
FROM (
  SELECT per_person.color,
    ANON_SUM(per_person.fruit_count, LN(3)/2) as number_eaten,
    ANON_COUNT(uid, LN(3)/2) as number_eaters
    FROM(
      SELECT * , ROW_NUMBER() OVER (
        PARTITION BY uid
        ORDER BY random()
      ) as row_num
      FROM (
        SELECT color, f.uid as uid, COUNT(fruit) as fruit_count
        FROM FruitEaten f INNER JOIN Shirts s ON (f.uid = s.uid)
        GROUP BY color, f.uid
      ) as per_person_raw
    ) as per_person
  WHERE per_person.row_num <= 5
  GROUP BY per_person.color
) as result
WHERE result.number_eaters > 50;
```

The restriction with this method of rewriting is that joins must not create any
rows of shared ownership. In the example above, this is satisfied.
