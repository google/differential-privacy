//
// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package com.google.privacy.differentialprivacy.testing;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Supplier;

/** Utility class providing voting mechanisms to reduce the error of randomized tests. */
public final class VotingUtil {

  private VotingUtil() {}

  /**
   * Casts a predefined number of votes to determine a majority. Stops early as soon as the majority
   * is clear.
   *
   * <p>Let n be the smallest odd integer greater or equal to {@code numberOfVotes}. This method is
   * intended to reduce the probability of error p of a randomized test by running the test n times
   * and taking the majority vote of the results. If the individual tests are independent, the
   * majority vote has a probability of error equal to F(floor(n/2)) where F is the CDF of a
   * binomial distribution over n trials with a success probability of 1 - p.
   *
   * @param voteGenerator an interface to supply the results of a randomized test. The results are
   *     interpreted as accept votes if true and as a reject votes otherwise.
   * @param numberOfVotes the number of eligible votes. The method ends early as soon as a majority
   *     is reached, i.e. not all votes may be cast. Moreover, in the case of an even number of
   *     votes a tie breaking vote may be cast if necessary. Note that this parameter must be
   *     positive.
   * @return true if the majority of votes are accept, false otherwise.
   */
  public static boolean runBallot(Supplier<Boolean> voteGenerator, int numberOfVotes) {
    checkArgument(numberOfVotes > 0, "The number of votes must be positive");
    int acceptVotes = 0;
    int rejectVotes = 0;
    while (acceptVotes <= numberOfVotes / 2 && rejectVotes <= numberOfVotes / 2) {
      if (voteGenerator.get()) {
        acceptVotes++;
      } else {
        rejectVotes++;
      }
    }
    return acceptVotes > rejectVotes;
  }
}
