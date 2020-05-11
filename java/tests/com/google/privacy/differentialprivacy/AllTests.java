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

/**
 * Provides a list of JUnit test classes to Bazel.
 * When creating a new test class, add it here.
 * {@see https://stackoverflow.com/questions/46365464/how-to-run-all-tests-in-bazel-from-a-single-java-test-rule}
 * for more details.
 */
@RunWith(Suite.class)
@SuiteClasses({
    BoundedSumBuilderTest.class,
    BoundedSumTest.class,
    CountBuilderTest.class,
    CountTest.class,
    GaussianNoiseTest.class,
    LaplaceNoiseTest.class,
})
public class AllTests {}



