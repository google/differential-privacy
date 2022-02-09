//
// Copyright 2022 Google LLC
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

import java.util.Random;

/**
 * Factory for creating Noise instances with a specified randomness source. This should only be used
 * when a large number of samples need to be generated without concern for the security of the
 * results as in the statistical tests.
 */
public final class TestNoiseFactory {
  private TestNoiseFactory() {}

  public static GaussianNoise createGaussianNoise(Random random) {
    return GaussianNoise.createForTesting(random);
  }

  public static LaplaceNoise createLaplaceNoise(Random random) {
    return LaplaceNoise.createForTesting(random);
  }
}
