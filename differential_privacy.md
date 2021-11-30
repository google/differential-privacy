# Differential privacy

Differential privacy offers a tradeoff between the accuracy of aggregations over
statistical databases (for example, mean) and the chance of learning something about
individual records in the database. This tradeoff is an easily configured
parameter; you can increase privacy by decreasing the accuracy of your statistics
(or vice versa). Unlike other anonymization schemes (such as k-anonymity) that
completely fail once too much data is released, differential privacy degrades
slowly when more data is released.

You can find a very high-level, non-technical introduction to differential
privacy in this
[blog post](https://desfontain.es/privacy/differential-privacy-awesomeness.html),
and a more detailed explanation of how it works in the book,
[The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
(linked as a PDF).

This library provides a collection of algorithms for computing differentially
private statistics over data. The algorithms are designed to require little
fancy mathematical knowledge to use; all the math is bundled into them.

# Key definitions
*A partition* is a subset of the data corresponding to a given value of the
aggregation criterion. Usually we want to aggregate each partition separately.
For example, if we count visits to restaurants, the visits for one particular
restaurant are a single partition, and the count of visits to that restaurant
would be the aggregate for that partition.

*A privacy unit* is an entity that we’re trying to protect with differential
privacy. Often, this refers to a single individual. An example of a more complex
privacy unit is a person+restaurant pair, which protects all visits by an
individual to a particular restaurant or, in other words, the fact that a
particular person visited any particular restaurant.

*Contribution bounding* is a process of limiting contributions by a single
individual (or an entity represented by a privacy key) to the output dataset
or its partition. This is key for DP algorithms, since protecting unbounded
contributions would require adding infinite noise.
