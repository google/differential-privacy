This example demonstrates how to compute differentially private statistics on a
[Netflix dataset](https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data).
To speed up calculations, we'll use a smaller sample of the full dataset.

The example code expects a CSV file in the following format: `movie_id`,
`user_id`, `rating`, `date`.

Using this data, the library computes these statistics:

*   Number of views of a certain movie (`count` metric)
*   Sum of all ratings of a certain movie (`sum` metric)
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
