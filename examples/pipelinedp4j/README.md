# Running code walkthrough of BeamExample

Begin by reviewing the [Problem statement](#problem-statement) to understand the
statistics we aim to calculate and the data involved.

Next, explore the following options for building and running the example:

*   [Using Bazel with the library source files](#running-using-bazel-and-library-sources)
*   [Using Maven with the library loaded from Maven repository as dependency](#running-using-maven)
*   [Using Maven on Google Cloud Platform (GCP)](#running-on-google-cloud-platform)

Finally, delve into the [code walkthrough](#code-walkthrough) for a
comprehensive understanding of how PipelineDP4j was employed to solve the task.

## Problem statement

This example demonstrates how to compute differentially private statistics on a
[Netflix dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
To speed up calculations, we'll use a smaller sample of the full dataset.

The example code expects a CSV file in the following format: `movie_id`,
`user_id`, `rating`, `date`.

Using this data, we want to compute the following statistics:

*   Number of views of a certain movie (`count` metric)
*   Number of users who watched a certain movie (`privacy_id_count` metric)
*   Average rating of a certain movie (`mean` metric)

The output is a TXT file in this format:

```
movieId=<value>, numberOfViews=<value>, averageOfRatings=<value>
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

Here's are the steps to build and run the example assuming you are in the
`examples/pipelinedp4j` directory:

1.  Build the program:

    ```shell
    bazelisk build ...
    ```

1.  Run the program:

    ```shell
    bazel-bin/src/main/java/com/google/privacy/differentialprivacy/pipelinedp4j/examples/BeamExample --inputFilePath=netflix_data.csv --outputFilePath=output.txt
    ```

1.  View the results:

    ```shell
    cat output.txt
    ```

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

Once Maven is installed, navigate to the `examples/pipelinedp4j` directory and
execute the following command:

```shell
mvn compile exec:java -Dexec.mainClass=com.google.privacy.differentialprivacy.pipelinedp4j.examples.BeamExample -Dexec.args="--inputFilePath=netflix_data.csv --outputFilePath=output.txt"
```

This command compiles the code and runs the `BeamExample` class with specified
input and output file paths.

To view the results, simply run:

```shell
cat output.txt
```

#### Running on Google Cloud Platform

This section explains how to run the Maven-built example on Google Cloud
Platform (GCP).

If you already have a Beam pipeline on GCP and want to make it differentially
private, simply update your `pom.xml` file (refer to the "PipelineDP4j
dependencies" section in the example `pom.xml` file):

1.  Add a dependency on PipelineDP4j.

1.  If your project isn't in Kotlin, include `kotlin-stdlib` as well.

After updating your `pom.xml`, modify your code to utilize PipelineDP4j (see the
[code walkthrough](#code-walkthrough) for guidance).

For those new to running Beam pipelines on GCP, follow these steps:

1.  Familiarize yourself with the
    [official example](https://cloud.google.com/dataflow/docs/quickstarts/create-pipeline-java).

1.  Prepare the input file:

    *   Create an `inputs` directory in your Cloud Storage bucket.
    *   Upload `netflix_data.csv` to this directory.

1.  Upload the `examples/pipelinedp4j` folder to your project using Cloud Shell
    (click ":" -> "Upload"). You can upload it fully but you only need `pom.xml`
    file and `src` folder.

1.  Navigate to the uploaded directory in Cloud Shell:

    ```shell
    cd pipelinedp4j
    ```

1.  Modify the mvn command from the official example:

    *   Add `inputFilePath` and `outputFilePath` arguments.
    *   Remove the `output` argument.
    *   Update the `mainClass` to
        `com.google.privacy.differentialprivacy.pipelinedp4j.examples.BeamExample`.

    Here's an example of the modified command (don't forget to set
    `<project_name>` and `<bucket_name>`):

    ```shell
    mvn -Pdataflow-runner compile exec:java -Dexec.mainClass=com.google.privacy.differentialprivacy.pipelinedp4j.examples.BeamExample -Dexec.args="--project=<project_name> --gcpTempLocation=gs://<bucket_name>/temp/ --inputFilePath=gs://<bucket_name>/inputs/netflix_data.csv --outputFilePath=gs://<bucket_name>/results/output.txt --runner=DataflowRunner --region=us-central1" -Pdataflow-runner
    ```

Running this command schedules the execution on GCP. You can then inspect the
job and its results as described in the official example.

## Code walkthrough

Let's deep into details how code for computing DP statistics is organized.

Warning: this API is experimental and will change in 2025 without backward
compatibility. The new version API released in 2024 will be long-term supported.

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

We need to read and preprocess data to `PCollection`, such that we can extract
from records Privacy Id, Group Key and Values to aggregate. In the example that
is encapsulated in the `readData` function.

```java
PCollection<MovieView> data = readData(pipeline);
```

### Create DP query

By creating DP query, we specify what DP operation on what data should be
computed.

In PipelineDP4j semantics of data is specified with data_extractors, functions
that take single dataset record and return corresponding object. There are 3
types of extractors: privacyIdExtractor, groupKeyExtractor, valueExtractor.

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

1.  Create `QueryBuilder` from data and privacyIdExtractor.

1.  Call `groupBy`: specify group by key, setting contribution bounding
    parameters and setting public groups if any. If no public group are
    specified, groups will be determined with the DP group selection procedure.

1.  Specify aggregations to compute (`count`, `countDistinctPrivacyUnits`,
    `sum`, `mean`, `variance`). For non-count aggregation it's required to
    specify a value (with valueExtractor) to aggregate.

1.  Finish building with call `.build()`.

Note that optionally it's possible to specify a DP budget per aggregation or per
`groupBy`. If the budget is not specified, the total budget will be split evenly
among all aggregations.

### Run query

On the running of the query we specify the total (&epsilon;, &delta;)$-DP budget
and DP mechanism to apply (Laplace mechanism in this case).

```java
PCollection<QueryPerGroupResult> result =
    query.run(new TotalBudget(/* epsilon= */ 1.1, /* delta= */ 1e-10), NoiseKind.LAPLACE);
```

### Saving results

Differential privacy has a nice property to be safe under post-processing. So
it's ok to do any post-processing of the output.
