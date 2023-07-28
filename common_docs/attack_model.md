# Attack Model for the Differential Privacy Libraries

## TL;DR

This doc summarizes our assumptions and requirements for using the DP Building
Block Libraries in a safe way.  We assume that an attacker does not have direct
access to the raw user data, has limited abilities to inject data into the
dataset, and has limited visibility into the resources consumed by the DP
Libraries.

## Differentially Private Output

The DP Building Block Libraries provide differentially private output.  In
layperson's terms, this means that an attacker should only be able to get very
limited additional knowledge from the output of the DP library.  An upper bound
on the amount of knowledge obtained from the output is configurable via the DP
parameters epsilon and delta.

## Use Cases

There are two intended use-cases of the library:

*   **Single data releases:** This scenario is a one-off aggregation of the raw
    user data.  The DP Libraries are used to aggregate the data once, and the
    result is then shared with a wider audience.
*   **Periodic data releases:** Similar to the above, but the library is used
    periodically to aggregate the data, and the result is then shared with a
    wider audience periodically.  The client is responsible for calculating an
    appropriate privacy budget (epsilon, delta) so that user data that is used
    in multiple aggregations is accounted for correctly.

## Non-Goals of the DP Libraries

*   The DP library is designed to be used as a component in a higher-level
    framework that mitigates against privacy attacks.  It is not designed to be
    used directly.  For instance, the DP library does not have the notion of a
    user and hence cannot filter the dataset to reduce the number of
    contributions per user.
*   The DP Library is not designed for an interactive setting, e.g., allowing an
    untrusted analyst to perform arbitrary queries.

## Attack Model

We assume that the DP Library is executed on trusted compute nodes.  This means
that clients must trust the hardware and any process that is running on the same
node.  If an attacker can control any process on these nodes, then the attack
surface is much larger than just via the DP libraries, since the node has access
to the raw user data (whether via the network or on a hard disc).

We assume that the DP Library is executed in batch mode.  After every run, the
output is eventually published to a wider audience and accessible to the
attacker.

### Attacker's prior knowledge

The attacker does not have access to the raw user data, otherwise there is
nothing left to protect.

The attacker might have knowledge about a subset of the raw user data:

1.  The attacker could have contributed to the dataset themselves.
2.  The attacker could have prior knowledge of the raw values of a large number
    of contributions (including the case where the attacker knows all raw values
    except for a single user's contribution).
    1.  The attacker could even use some outside mechanism to learn about all
        the contributions.  However, from the output of the DP Libraries, the
        attacker should not be able to infer whether they learned about all the
        contributions or whether they only obtained a strict subset of the raw
        data source.  In particular, the attacker should not be able to infer
        the values that a user contributed to the dataset.

### Injecting malicious data

The attacker can forge a very large number or even all contributions to the raw
input data set.  However, from the output of the DP Libraries, it should be
impossible for the attacker to learn whether there were any other entries in the
dataset (i.e. whether all contributions were forged).

Note: Depending on the application logic, there can be some mitigations against
malicious data, e.g., applying rounding and/or enforcing a typical number of
contributions per privacy unit.

### Order of events

The attacker can control the sequence in which user data is passed to the DP
Library, including the case where this data is malicious and
attacker-controlled.  The DP Library protects against attacks that target the
non-associativity of floating point arithmetic and similar attacks using the
order of events.

### Side channels

The execution is hidden from the attacker.  In particular, we assume that the
attacker does not have additional information about:

1.  How the raw user data is retrieved from the data storage.
2.  How processes are executed, their memory consumption, CPU utilization,
    network usage, or timing information.
3.  The state of the random number generator and the amount of entropy
    available.
