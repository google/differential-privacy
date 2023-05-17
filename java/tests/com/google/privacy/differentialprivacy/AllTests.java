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

package com.google.privacy.differentialprivacy;

import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/** Provides a list of JUnit test classes to Bazel. When creating a new test class, add it here. */
@RunWith(Suite.class)
@SuiteClasses({
  ApproximateBoundsBuilderTest.class,
  ApproximateBoundsTest.class,
  BoundedMeanBuilderTest.class,
  BoundedMeanConfidenceIntervalTest.class,
  BoundedMeanTest.class,
  BoundedQuantilesTest.class,
  BoundedQuantilesBuilderTest.class,
  BoundedQuantilesConfidenceIntervalTest.class,
  BoundedSumBiasTest.class,
  BoundedSumBuilderTest.class,
  BoundedSumConfidenceIntervalTest.class,
  BoundedSumTest.class,
  BoundedVarianceBuilderTest.class,
  BoundedVarianceTest.class,
  CountBiasTest.class,
  CountBuilderTest.class,
  CountConfidenceIntervalTest.class,
  CountTest.class,
  GaussianNoiseConfidenceIntervalTest.class,
  GaussianNoiseQuantileTest.class,
  GaussianNoiseTest.class,
  LaplaceNoiseConfidenceIntervalTest.class,
  LaplaceNoiseQuantileTest.class,
  LaplaceNoiseTest.class,
  LongBoundedSumBuilderTest.class,
  LongBoundedSumBiasTest.class,
  LongBoundedSumConfidenceIntervalTest.class,
  LongBoundedSumTest.class,
  SamplingUtilTest.class,
  SecureNoiseMathTest.class,
})
public class AllTests {}
