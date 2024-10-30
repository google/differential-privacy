# Running code walkthrough of BeamExample.

## Running

This example demonstrates how to compute differentially private statistics on a
[Netflix dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
To speed up calculations, we'll use a smaller sample of the full dataset.

The example code expects a CSV file in the following format: `movie_id`,
`user_id`, `rating`, `date`.

Using this data, the library computes these statistics:

*   Number of views of a certain movie (`count` metric)
*   Number of users who watched a certain movie (`privacy_id_count` metric)
*   Average rating of a certain movie (`mean` metric)

The output is a TXT file in this format:

```
(movieId=<value>, numberOfViews=<value>, sumOfRatings=<value>)
```

The entries will be sorted by `movie_id`. The counts are not rounded to the
nearest integers. You can do this yourself if you want.

Here's are the steps to run the example:

1.  Go to the example directory:

    ```shell
    cd examples/kotlin
    ```

1.  Download the
    [Netflix Prize data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
    and extract `combined_data_2.txt` into the `examples/kotlin` directory.

1.  Create a sample dataset:

    ```shell
    awk -v OFS=',' '/^[0-9]+:$/ {movie_id=substr($1, 1, length($1)-1)} /^[0-9]+,[0-9]+/ {print movie_id, $0}' combined_data_2.txt | \
    head -n 10000 > netflix_data.csv
    ```

    This command takes the first 10,000 lines from `combined_data_2.txt`,
    reformats them into the expected format, and saves them in
    `netflix_data.csv`.

1.  Build the program:

    ```shell
    bazel build ...
    ```

1.  Run the program:

    ```shell
    bazel-bin/BeamExample --local-input-file-path="./netflix_data.csv" --local-output-file-path="./output.txt"
    ```

1.  View the results:

    ```shell
    cat output.txt
    ```

## Code walkthrough
Let's deep into details how code for computing DP statistics is organized.

Warning: this API is experimental and will change in 2025 without backward 
compatibility. The new version API released in 2024 will be long-term supported.

### Key definitions:

- **(Privacy) budget**: every operation leaks some information about individuals. The total privacy cost of a pipeline is the sum of the costs of calculated statistics. You want this to be below a certain total cost. That's your budget. Typically, the greek letters 'epsilon' and 'delta' (&epsilon; and &delta;) are used to define the budget.
Bigger epsilon => more budget => less privacy.

- **Group:** a group is a subset of the data corresponding to a given value of the aggregation criterion. In our example, the groups are movies.

- **Group key:** this is the group identifier. Since in our example the data are aggregated per movie, the group key is a movie_id.

- **A privacy unit** is an entity that we’re trying to protect with differential privacy. Often, this refers to a single individual. An example of a more complex privacy unit is a person+restaurant pair, which protects all visits by an individual to a particular restaurant or, in other words, the fact that a particular person visited any particular restaurant.

- **Privacy ID:** an ID of the unit of privacy that we are protecting. For example, if we protect the presence of the user in a dataset, the privacy ID is the user ID. In this example, the privacy ID is a user ID who watched a movie.

- **Contribution bounding** is a process of limiting contributions by a single individual (or an entity represented by a privacy key) to the output dataset or its partition. This is key for DP algorithms, since protecting unbounded contributions would require adding infinite noise.

- **Group selection** is a process of identifying the partition keys that are safe to release in the sense that they don’t break the DP guarantees and don’t leak any user information.

- **Public groups** are partition keys that are publicly known and hence don’t leak any user information.
In our case we will use public groups since our groups are movies and they are publicly known.

### Reading and pre-processing data
We need to read and preprocess data to `PCollection`, such that we can extract from records Privacy Id, Group Key and Values to aggregate.
In the example that is encapsulated in the `readData` function.

```java
PCollection<MovieView> data = readData(pipeline);
```

### Create DP query

By creating DP query, we specify what DP operation on what data should be computed.

In PipelineDP4j semantics of data is specified with data_extractors, functions that take single dataset record and return corresponding object.
There are 3 types of extractors: privacyIdExtractor, groupKeyExtractor, valueExtractor.

```java
    var query =
        QueryBuilder.from(data, /* privacyIdExtractor= */ new UserIdExtractor())
            .groupBy(
                /* groupKeyExtractor= */ new MovieIdExtractor(),
                /* maxGroupsContributed= */ 3,
                /* maxContributionsPerGroup= */ 1,
                usePublicGroups ? publiclyKnownMovieIds(pipeline) : null)
            .countDistinctPrivacyUnits("numberOfViewers")
            .count(/* outputColumnName= */ "numberOfViews")
            .mean(
                new RatingExtractor(),
                /* minValue= */ 1.0,
                /* maxValue= */ 5.0,
                /* outputColumnName= */ "averageOfRatings",
                /* budget= */ null)
            .build();
```

Building of query is similar to writing SQL query. It consists:

1. Create `QueryBuilder` from data and privacyIdExtractor.

1. Call `groupBy`: specify group by key, setting contribution bounding parameters and setting public groups if any.
If no public group are specified, groups will be determined with the DP group selection procedure.

1. Specify aggregations to compute (`count`, `countDistinctPrivacyUnits`, `sum`, `mean`, `variance`).
For non-count aggregation it's required to specify a value (with valueExtractor) to aggregate.

1. Finish building with call `.build()`.

Note that optionally it's possible to specify a DP budget per aggregation or
per `groupBy`. If the budget is not specified, the total budget will be split evenly among all
aggregations.

### Run query

On the running of the query we specify the total (&epsilon;, &delta;)$-DP budget and DP mechanism to apply (Laplace mechanism in this case).

```java
    PCollection<QueryPerGroupResult> result =
        query.run(new TotalBudget(/* epsilon= */ 1.1, /* delta= */ 1e-10), NoiseKind.LAPLACE);

```

### Saving results
Differential privacy has a nice property to be safe under post-processing.
So it's ok to do any post-processing of the output.
