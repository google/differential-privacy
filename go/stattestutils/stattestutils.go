//
// Copyright 2023 Google LLC
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

// Package stattestutils provides basic statistical utility functions.
//
// This package is not optimized for performance or speed and is only intended
// to be used in tests.
package stattestutils

import "math"

// SampleMean returns the mean of a slice, calculated as the average over the
// values in the slice.
func SampleMean(values []float64) float64 {
	var sum float64 = 0.0
	for _, v := range values {
		sum += v
	}
	return sum / math.Max(1, float64(len(values)))
}

// SampleVariance returns the variance of a slice, calculated as the sum of
// sqaures of the distance to the mean of each of the values, divided by the
// number of values.
func SampleVariance(values []float64) float64 {
	mean := SampleMean(values)
	var sumOfSquares float64 = 0.0
	for _, v := range values {
		sumOfSquares += math.Pow(v-mean, 2)
	}
	return sumOfSquares / math.Max(1, float64(len(values)))
}
