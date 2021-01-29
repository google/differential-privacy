# Example using Differential Privacy library

In this directory, we give a simple example of how to use the C++ Differential
Privacy library.

## Zoo Animals

There are around 200 animals at Farmer Fred's zoo. Every day, Farmer Fred feeds
the animals as many carrots as they desire. The animals record how many carrots
they have eaten per day. For this particular day, the number of carrots eaten
can be seen in `animals_and_carrots.csv`.

At the end of each day, Farmer Fred often asks aggregate question about how many
carrots everyone ate. For example, he wants to know how many carrots are eaten
each day, so he knows how many to order the next day. The animals are fearful
that Fred will use the data against their best interest. For example, Fred could
get rid of the animals who eat the most carrots!

To protect themselves, the animals decide to use the C++ Differential Privacy
library to aggregate their data before reporting it to Fred. This way, the
animals can control the risk that Fred will identify individuals' data while
maintaining an adequate level of accuracy so that Fred can continue to run the
zoo effectively.

The animals have implemented a CarrotReporter tool in `animals_and_carrots.h` to
obtain DP aggregate data to report to Fred. We document one of these reports in
`report_the_carrots.cc`.

## Data

Each row in `animals_and_carrots.csv` is composed of the name of an animal, and
the number of carrots it has eaten, comma-separated.

## Per-animal Privacy

Notice that each animal owns at most one row in the data. This means that we
provide per-animal privacy. Suppose that some animal appears multiple times in
the csv file. That animal would own more than one row in the data. In this case,
using this DP library would not guarantee per-animal privacy! The animals would
first have to pre-process their data in a way such that each animal doesn't own
more than one row.

## Privacy budget

If Farmer Fred continues to query the carrot data many times in a given day,
then he can draw conclusions about the data that breaks privacy. For example,
asking about the mean number of carrots eaten repeatedly would yield a
distribution centered about the true mean. To prevent this, the animals have
implemented a privacy budget tracking feature in CarrotReporter. The budget is
a fraction; at the beginning of the day, the budget is 1. Each time the animals
answer one of Fred's questions, the privacy budget decreases. When the privacy
budget is no longer positive, the animals refuse to answer any more of Fred's
queries.

## How to Run
```shell
$ cd examples/cc
$ bazel run report_the_carrots
```

