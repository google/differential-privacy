# Running code walkthrough of the examples

Begin by reviewing the [Problem statement](#problem-statement) to understand the
statistics we aim to calculate and the data involved.

Next, explore the following options for building and running the example:

*   [Using Bazel with the library source files](#running-using-bazel-and-library-sources)
*   [Using Maven with the library loaded from Maven repository as dependency](#running-using-maven)
*   [Using Dataflow (Beam) on Google Cloud Platform (GCP)](#running-on-dataflow-beam)
*   [Using Dataproc (Spark) on Google Cloud Platform (GCP)](#running-on-dataproc-spark)

Finally, delve into the [code walkthrough](#code-walkthrough) for a
comprehensive understanding of how PipelineDP4j was employed to solve the task.

## Problem statement

This example demonstrates how to compute differentially private statistics on a
[Netflix dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
To speed up calculations, we'll use a smaller sample of the full dataset.

The example code expects a CSV file in the following format: `movie_id`,
`user_id`, `rating`, `date`.

Using this data, we want to compute the following statistics:

*   Number of users who watched a certain movie (`privacy_id_count` metric)
*   Number of views of a certain movie (`count` metric)
*   Average rating of a certain movie (`mean` metric)

For column-based DataFrame API the output is a CSV file in the following format:

```
movieId, numberOfViewers, numberOfViews, averageOfRatings
value, value, value, value
value, value, value, value
...
```

For row-based API then the output will be in the following format:

```
movieId=<value>, numberOfViewers=<value>, numberOfViews=<value>, averageOfRatings=<value>
movieId=<value>, numberOfViewers=<value>, numberOfViews=<value>, averageOfRatings=<value>
...
```

## Running

Before we run the example, we need to prepare the input dataset.

1.  Download the
    [Netflix Prize data](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data)
    and extract `combined_data_2.txt`.

1.  Create a sample dataset:

    ```shell
    awk -v OFS=',' '/^[0-9]+:$/ {movie_id=substr($1, 1, length($1)-1)} /^[0-9]+,[0-9]+/ {print movie_id, $0}' combined_data_2.txt | \
    head -n 10000 > netflix_data.csv
    ```

    This command takes the first 10,000 lines from `combined_data_2.txt`,
    reformats them into the expected format, and saves them in
    `netflix_data.csv`.

1.  Go to the example directory:

    ```shell
    cd examples/pipelinedp4j
    ```

1.  Copy the prepared `netflix_data.csv` file:

    ```shell
    cp path/to/netflix_data.csv netflix_data.csv
    ```

    If you want to run on GCP, then you will need to copy it to your bucket.

### Running using Bazel and library sources

To build and run this way you need the source files of the library. It means you
are supposed to have both `pipelinedp4j` and `examples/pipelinedp4j` directories
at your root directory. You can achieve that by just
[cloning the whole GitHub repository](https://github.com/google/differential-privacy/tree/main?tab=readme-ov-file#how-to-build).

The build will be performed using Bazel, i.e. you need to have bazelisk
installed. See the
[repository root REAME](https://github.com/google/differential-privacy/tree/main?tab=readme-ov-file#how-to-build)
on how to install it.

`pom.xml` file is not necessary for this build. We need it to build with Maven.

Here are the steps to build and run the example assuming you are in the
`examples/pipelinedp4j` directory:

1.  Build the code:

    ```shell
    bazelisk build ...
    ```

1.  Run the program (if you want to run Spark example, change `beam` to `spark`,
    `BeamExample` to `SparkExample` or `SparkDatasetExample` and
    `--outputFilePath=output.txt` to `--outputFolder=output`):

    ```shell
    bazel-bin/beam/src/main/java/com/google/privacy/differentialprivacy/pipelinedp4j/examples/BeamExample --inputFilePath=netflix_data.csv --outputFilePath=output.txt
    ```

1.  View the results.

    For Beam: `cat output.txt`

    For Spark the output is written to a folder and the result is stored in a
    file whose name starts with `part-00000`: `cat output/part-00000<...>`

### Running using Maven

This section describes how to run the example using Maven. This approach
utilizes a
[pre-compiled version of PipelineDP4j](https://mvnrepository.com/artifact/com.google.privacy.differentialprivacy.pipelinedp4j/pipelinedp4j)
from the Maven repository, eliminating the need for the library source files and
the Bazel files (`WORKSPACE.bazel`, `.bazelversion` and `BUILD.bazel`).

To proceed, ensure Maven is installed on your system. If you're using Linux or
MacOS, you can install it by running `sudo apt-get install maven` or `brew
install maven`, respectively. While any Maven version should work, refer to
[this documentation](https://www.baeldung.com/install-maven-on-windows-linux-mac)
for specific version requirements or Windows installation instructions.

Also, make sure that Maven uses Java <= 11: `mvn -v`. If not, install JDK <= 11
and update the `JAVA_HOME` accordingly. Otherwise you might have runtime errors
when running Spark example.

Once Maven is installed, navigate to the directory of a backend you want to use:

*   `examples/pipelinedp4j/beam` for Beam

*   `examples/pipelinedp4j/spark` for Spark

Then execute the following command with updated `inputFilePath` (if you want to
run on Spark, change `BeamExample` to `SparkExample` or `SparkDatasetExample`
and `--outputFilePath=output.txt` to `--outputFolder=output`):

```shell
mvn compile exec:java -Dexec.mainClass=com.google.privacy.differentialprivacy.pipelinedp4j.examples.BeamExample -Dexec.args="--inputFilePath=<absolute_paht_to>/netflix_data.csv --outputFilePath=output.txt"
```

This command compiles the code and runs the example with specified input and
output file paths.

View the results.

For Beam `cat output.txt` For Spark the output is written to a folder and the
result is stored in a file whose name starts with `part-00000`: `cat
output/part-00000<...>`

### Running on Google Cloud Platform

This section explains the examples on Google Cloud Platform (GCP).

#### Running on Dataflow (Beam)

If you already have a Beam pipeline on GCP and want to make it differentially
private, simply update your `pom.xml` file (refer to the "PipelineDP4j
dependencies" section in the root `pom.xml` file):

1.  Add a dependency on PipelineDP4j.

1.  If your project isn't in Kotlin, include `kotlin-stdlib` as well.

After updating your `pom.xml`, modify your code to utilize PipelineDP4j (see the
[code walkthrough](#code-walkthrough) for guidance).

For those new to running Beam pipelines on GCP, follow these steps:

1.  Familiarize yourself with the
    [official example](https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-java).

1.  Upload the `examples/pipelinedp4j` folder to your project using Cloud Shell
    (click ":" -> "Upload").

1.  Navigate to the uploaded directory in Cloud Shell and build the example:

    ```shell
    cd pipelinedp4j && mvn clean install
    ```

1.  Go to `beam` folder and then run the modified mvn command from the official
    example:

    *   Add `inputFilePath` and `outputFilePath` arguments.
    *   Remove the `output` argument.
    *   Update the `mainClass` to
        `com.google.privacy.differentialprivacy.pipelinedp4j.examples.BeamExample`.

    Here's an example of the modified command (don't forget to set
    `<project_name>` and `<bucket_name>`):

    ```shell
    cd beam && mvn -Pdataflow-runner compile exec:java -Dexec.mainClass=com.google.privacy.differentialprivacy.pipelinedp4j.examples.BeamExample -Dexec.args="--project=<project_name> --gcpTempLocation=gs://<bucket_name>/temp/ --inputFilePath=gs://<bucket_name>/netflix_data.csv --outputFilePath=gs://<bucket_name>/output.txt --runner=DataflowRunner --region=us-central1" -Pdataflow-runner
    ```

Running this command schedules the execution on GCP. You can then inspect the
job and its results as described in the official example.

#### Running on Dataproc (Spark)

In the guide below if you want to use SparkDataFrameExample, just replace all
occurences of SparkExample with SparkDataFrameExample.

If you already have a Spark pipeline then just add a PipelineDP4j dependency in
your Maven build, the same way as it described for Beam above.

If you are new to Spark on GCP, then go through the
[official Spark on GCP documentation](https://cloud.google.com/dataproc-serverless/docs/quickstarts/spark-batch)
to ensure your environment is correctly set up. Then do the following.

1.  Upload the example sources as in the Beam instructions above.

1.  Build a deploy jar (also called Uber jar):

    ```shell
    mvn clean install && cd spark && mvn package assembly:single
    ```

1.  Submit the Spark job:

    ```shell
    gcloud dataproc batches submit spark --region=us-central1 --jars=./target/spark-1.0-SNAPSHOT-jar-with-dependencies.jar --class=com.google.privacy.differentialprivacy.pipelinedp4j.examples.SparkExample --deps-bucket=gs://<bucket_name> -- --inputFilePath=gs://<bucket_name>/netflix_data.csv --outputFolder=gs://<bucket_name>/output
    ```

After finish you can inspect the result on GCP bucket.

## Code walkthrough

Let's deep into details how code for computing DP statistics is organized.

### Key definitions:

-   **(Privacy) budget**: every operation leaks some information about
    individuals. The total privacy cost of a pipeline is the sum of the costs of
    calculated statistics. You want this to be below a certain total cost.
    That's your budget. Typically, the greek letters 'epsilon' and 'delta'
    (&epsilon; and &delta;) are used to define the budget. Bigger epsilon =>
    more budget => less privacy.

-   **Group:** a group is a subset of the data corresponding to a given value of
    the aggregation criterion. In our example, the groups are movies.

-   **Group key:** this is the group identifier. Since in our example the data
    are aggregated per movie, the group key is a movie_id.

-   **A privacy unit** is an entity that we’re trying to protect with
    differential privacy. Often, this refers to a single individual. An example
    of a more complex privacy unit is a person+restaurant pair, which protects
    all visits by an individual to a particular restaurant or, in other words,
    the fact that a particular person visited any particular restaurant.

-   **Privacy ID:** an ID of the unit of privacy that we are protecting. For
    example, if we protect the presence of the user in a dataset, the privacy ID
    is the user ID. In this example, the privacy ID is a user ID who watched a
    movie.

-   **Contribution bounding** is a process of limiting contributions by a single
    individual (or an entity represented by a privacy key) to the output dataset
    or its partition. This is key for DP algorithms, since protecting unbounded
    contributions would require adding infinite noise.

-   **Group selection** is a process of identifying the partition keys that are
    safe to release in the sense that they don’t break the DP guarantees and
    don’t leak any user information.

-   **Public groups** are partition keys that are publicly known and hence don’t
    leak any user information. In our case we will use public groups since our
    groups are movies and they are publicly known.

### Reading and pre-processing data

We need to read and preprocess data to a distributed collection (e.g.
`PCollection` in Beam or `Dataset` in Spark), such that we can extract from
records Privacy Id, Group Key and Values to aggregate. In the example that is
encapsulated in the `readData` function.

```java
Dataset<MovieView> data = readData(spark);
```

### Create DP query

By creating DP query, we specify what DP operation on what data should be
computed.

In PipelineDP4j there are two ways to represent the data. Column-based and
row-based. Column-based are the APIs that expect dataframes as input and you
specify data semantics via column names. Row-based are the APIs that expect
distributed collections as input and you specify data semantics via
data_extractors - functions that take single dataset record (row) and return
corresponding object. There are 3 types of extractors: privacyIdExtractor,
groupKeyExtractor, valueExtractor. In the column-based API instead of extactors
you have to provide list of column names.

```java
var groupsType =
    usePublicGroups
        ? GroupsType.PublicGroups.create(publiclyKnownMovieIds(spark))
        : new GroupsType.PrivateGroups();
var query =
    SparkQueryBuilder.from(
            data,
            /* privacyUnitExtractor= */ MovieView::getUserId,
            new ContributionBoundingLevel.DATASET_LEVEL(
                /* maxGroupsContributed= */ 3, /* maxContributionsPerGroup= */ 1))
        .groupBy(/* groupKeyExtractor= */ MovieView::getMovieId, groupsType)
        .countDistinctPrivacyUnits(/* outputColumnName= */ "numberOfViewers")
        .count(/* outputColumnName= */ "numberOfViews")
        .aggregateValue(
            /* valueExtractor= */ new RatingExtractor(),
            /* valueAggregations= */ new ValueAggregationsBuilder()
                .mean(/* outputColumnName= */ "averageOfRatings"),
            /* contributionBounds= */ new ContributionBounds(
                /* totalValueBounds= */ null,
                /* valueBounds= */ new Bounds(/* minValue= */ 1.0, /* maxValue= */ 5.0)))
        .build(new TotalBudget(/* epsilon= */ 1.1, /* delta= */ 1e-10), NoiseKind.LAPLACE);
```

Building of query is similar to writing SQL query. It consists:

1.  Create `SparkQueryBuilder` from data and privacyUnitExtractor, indicating on
    what level contributions from a privacy unit should be bounded (see KDoc for
    detailed description of available levels). There are also other builders:
    `BeamQueryBuilder`, `SparkDataFrameQueryBuilder` and `LocalQueryBuilder`.
    For `SparkDataFrameQueryBuilder` you have to specify column names instead of
    the extractor.

1.  Call `groupBy`: specify keys to group by via groupKeyExtractor or
    groupKeyColumnNames and groups type (either public or private). If groups
    are public, you will need to explicitly provide them, if private then the
    groups will be determined with the DP group selection procedure.

1.  Specify aggregations to compute (`count`, `countDistinctPrivacyUnits`,
    `sum`, `mean`, `variance`). For non-count aggregation it's required to
    specify a value (with valueExtractor or valueColumnName) to aggregate.

1.  Finish building with call `.build(totalBudget, noiseType)`. On the building
    of the query we specify the total (&epsilon;, &delta;)-DP budget and DP
    mechanism to apply (Laplace mechanism in this case).

Note that optionally it's possible to specify a DP budget per aggregation or per
`groupBy`. If the budget is not specified, the total budget will be split evenly
among all aggregations without explicit budgeting.

### Run query

```java
Dataset<QueryPerGroupResult> result = query.run();
```

The result will contain a collection of `QueryPerGroupResult`s which consist of
a group key and mapping from output column names to calculated metrics.

If you used the column-based API then you will get a dataframe back (e.g.
`Dataset<Row>` in Spark) which contains "group by key" columns and columns with
requested aggregations.

### Saving results

Differential privacy has a nice property to be safe under post-processing. So
it's ok to do any post-processing of the output.
