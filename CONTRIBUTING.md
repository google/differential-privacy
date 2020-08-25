# How to Contribute

We are happy to accept contributions to this project. Please follow these
guidelines when sending us a pull request.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted
one (even if it was for a different project), you probably don't need to do it
again.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google.com/conduct/).

## Does my new feature belong here?

We use the following principles to guide what we add to our libraries. If your
contribution doesn't align with these principles, we're likely to decline.

* **Simplicity over completeness:** the libraries should provide simple features
which are easy to use, such as noise generation and aggregation functions. To be
considered for inclusion into the library, a feature's usefulness will be
weighed against the technical complexity it adds to the library. Very basic
operations like noise addition should stay as simple as possible.
* **Simplicity over consistency:** given the language differences, we
acknowledge that the exact interfaces may differ between C++, Java, Python and
Go. We will try to keep the overall API and architecture consistent between the
languages, but will prioritize idiomatic interfaces over strict consistency.
* **Support for distributed calculations:** all functions requiring aggregations
must offer an API that lets them be used in large-scale parallel computing
frameworks.
* **Fine building blocks over large aggregates:** one should be able to
use sub-operations like noise generation and bounds approximation separately
from aggregation functions.
* **Unbiasedness:** aggregations should be unbiased if possible. In particular,
we prefer unbiased aggregations over aggregations that post process results for
consistency reasons (e.g. we do not clipp negative count values to 0 as this
would introduce bias). However, we may use biased aggregations if an unbiased
solution is not known, provides inferior utility, does not support distributed
computation or is significantly more complex to understand/implement/maintain.
The library should clearly indicate which aggregations are unbiased and test for
this property.
* **Robust Testing:** each feature must come with a full set of unit tests, and
the privacy guarantees must be tested end-to-end.
* Markdown is preferred for explaining complex concepts and math over lengthy
code documentation.

If you're uncertain whether a planned contribution fits with these principles,
[open an issue](https://github.com/google/differential-privacy/issues/new)
and describe what you want to add. We'll let you know whether it's something we
want to include and will help you figure out the best way to implement it.

## Guidelines for pull requests

* For large or speculative changes, or significant new functionality, please
[open an issue](https://github.com/google/differential-privacy/issues/new)
first and discuss your idea with us.
* If you are a Googler, it is preferable to create an internal CL and have
it reviewed and submitted. The code publication process will deliver the change
to GitHub. If you want to discuss your changes with us, please open an issue in
the GitHub project.
* Create small PRs that are narrowly focused on addressing a single concern.
Small PRs are easier to review, and less likely to see one change get held up
due to problems with an unrelated change in the same PR.
* Code should follow the
[Google style guide](https://google.github.io/styleguide/) in the relevant
language.
* Provide a good PR description as a record of what change is being made and
why. Link to a GitHub issue if it exists.
* Don't fix code style and formatting unless you are already changing that line
to address an issue. PRs with irrelevant changes won't be merged. If you do want
to fix formatting or style, do that in a separate PR.
* Unless your PR is trivial, you should expect there will be reviewer comments
that you'll need to address before merging. We expect you to be reasonably
responsive to those comments, otherwise the PR will be closed after 2-3 weeks of
inactivity.
* Maintain clean commit history and use meaningful commit messages. PRs with
messy commit history are difficult to review and won't be merged. Use
`rebase -i upstream/main` to curate your commit history and/or to bring in
latest changes from main (but avoid rebasing in the middle of a code review).
* Keep your PR up to date with upstream/head (if there are merge conflicts,
we can't really merge your change).
* All tests need to be passing before your change can be merged. We recommend
you run tests locally.
* All submissions, including submissions by project members, require review.
