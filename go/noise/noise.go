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

// Package noise contains methods to generate and add noise to data.
package noise

import (
	"math"

	log "github.com/golang/glog"
)

// Kind is an enum type. Its values are the supported noise distributions types
// for differential privacy operations.
type Kind int

// Noise distributions used to achieve Differential Privacy.
const (
	GaussianNoise Kind = iota
	LaplaceNoise
)

// ToNoise converts a Kind into a Noise instance.
func ToNoise(k Kind) Noise {
	switch k {
	case GaussianNoise:
		return Gaussian()
	case LaplaceNoise:
		return Laplace()
	default:
		log.Warningf("ToNoise: unknown kind (%v) specified", k)
	}
	return nil
}

// ToKind converts a Noise instance into a Kind.
func ToKind(n Noise) Kind {
	switch n {
	case Gaussian():
		return GaussianNoise
	case Laplace():
		return LaplaceNoise
	default:
		log.Warningf("ToKind: unknown Noise (%v) specified", n)
	}
	return GaussianNoise
}

// ConfidenceInterval holds lower and upper bounds as float64 for the confidence interval.
type ConfidenceInterval struct {
	LowerBound, UpperBound float64
}

// toConfidenceIntervalInt64 rounds the lower and upper bounds of a ConfidenceInterval struct for
// integer valued noise operations.
func (confInt ConfidenceInterval) toConfidenceIntervalInt64() ConfidenceInterval {
	return ConfidenceInterval{LowerBound: math.Round(confInt.LowerBound), UpperBound: math.Round(confInt.UpperBound)}
}

// Noise is an interface for primitives that add noise to data to make it differentially private.
type Noise interface {
	// AddNoiseInt64 noise to the specified int64 x so that the output is ε-differentially
	// private given the L_0 and L_∞ sensitivities of the database.
	AddNoiseInt64(x, l0sensitivity, lInfSensitivity int64, epsilon, delta float64) int64

	// AddNoiseFloat64 noise to the specified float64 x so that the output is ε-differentially
	// private given the L_0 and L_∞ sensitivities of the database.
	AddNoiseFloat64(x float64, l0sensitivity int64, lInfSensitivity, epsilon, delta float64) float64

	// Threshold returns the smallest threshold k needed in settings where the Noise instance
	// is used to achieve differential privacy on histograms where the inclusion of histogram
	// partitions depends on which privacy units are present in the database.
	//
	// Inputs:
	//   l0Sensitivity: The maximum number of partitions that a privacy unit can contribute to.
	//   lInfSensitivity: How much any single partition's value can change from
	//     the contribution of a single privacy unit. When adding a privacy unit results in the
	//     creation of a new partition, this bounds the magnitude of that partition.
	//   epsilon: The parameter ε passed to AddNoise.
	//   noiseDelta: The parameter δ passed to AddNoise.
	//   thresholdDelta: Differential privacy loss (0, delta) incurred by thresholding,
	//     i.e. the probability to output a partition that only has one privacy unit in it.
	//
	// More precisely, Threshold returns the smallest k such that the following algorithm:
	//
	//   func Histogram(records []struct{key string, value float64}) map[string]float64 {
	//     sums := make(map[string]float64)
	//	   for _, record := range records {
	//       sums[record.key] = sums[record.key]+record.value
	//     }
	//     noisySums := make(map[string]float64)
	//     for key, sum := range sums {
	//       noisySum := AddNoiseFloat64(sum, sensitivity, epsilon, noiseDelta)
	//       if noisySum ≥ k:
	//         noisySums[key] = noisySum
	//     }
	//     return noisySums
	//   }
	//
	// satisfies (epsilon,noiseDelta+thresholdDelta)-differential privacy under the
	// given assumptions of L_0 and L_∞ sensitivities.
	Threshold(l0Sensitivity int64, lInfSensitivity, epsilon, noiseDelta, thresholdDelta float64) float64

	// ConfidenceIntervalInt64 will return a ConfidenceInterval struct with the given
	// confidenceLevel, using the int64 noisedValue, l0Sensitivity, lInfSensitivity int64 and
	// epsilon, delta float64 for the distribution.
	ConfidenceIntervalInt64(noisedValue, l0Sensitivity, lInfSensitivity int64, epsilon, delta, confidenceLevel float64) (ConfidenceInterval, error)

	// ConfidenceIntervalFloat64 will return a ConfidenceInterval struct with the given
	// confidenceLevel, using the float64 noisedValue, l0Sensitivity int64 and lInfSensitivity,
	// epsilon, delta float64 for the distribution.
	ConfidenceIntervalFloat64(noisedValue float64, l0Sensitivity int64, lInfSensitivity, epsilon, delta, confidenceLevel float64) (ConfidenceInterval, error)
}
