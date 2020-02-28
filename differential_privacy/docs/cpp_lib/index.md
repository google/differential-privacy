

<!-- This file is auto-generated. Do not edit. -->

# Differential Privacy

Differential privacy offers a tradeoff between the accuracy of aggregations over
statistical databases (e.g. mean) and the chance of learning something about
individual records in the database. This tradeoff is an easily configured
parameter; you can increase privacy by decreasing the accuracy of your
statistics (or vice versa). Unlike other anonymization schemes (such as
k-anonymity) that completely fail once too much data is released, differential
privacy degrades slowly when more data is released.

You can find a very high-level, non-technical introduction to differential
privacy in this
[blog post](https://desfontain.es/privacy/differential-privacy-awesomeness.html),
and a more detailed explanation of how it works in this
[longer article](https://github.com/frankmcsherry/blog/blob/master/posts/2016-02-06.md).

This library provides a collection of algorithms for computing differentially
private statistics over data. The algorithms are designed to require little
fancy mathematical knowledge to use; all the math is bundled into them.

All of our algorithms inherit from the
[`Algorithm`](https://github.com/google/differential-privacy/blob/master/differential_privacy/docs/cpp_lib/algorithms/algorithm.md) base class; you can
find more details about the API there.

Here's a minimal example showing how to compute the count of some data:

```
#include "differential_privacy/algorithms/count.h"

// Epsilon is a configurable parameter. A lower value means more privacy but
// less accuracy.
double count(const vector<double>& vals, double epsilon) {
  // Construct the Count object to run on double inputs.
  std::unique_pointer<differential_privacy::Count<double>> count =
     differential_privacy::Count<double>::Builder().SetEpsilon(epsilon)
                                                   .Build()
                                                   .ValueOrDie();

  // Compute the count and get the result.
  differential_privacy::Output result = count->Result(v.begin(), v.end());

  // GetValue can be used to extract the value from an Output (or Input)
  // protobuf. For count, this is always an int value.
  return differential_privacy::GetValue<int>(result);
}

```

## Caveats

All of our algorithms assume one input element per user. All of a user's
contributions should be reduced to a single input element before being passed to
our algorithms.

