//
// Copyright 2021 Google LLC
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

package dpagg

import (
	"fmt"
	"math"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/noise"
)

// BoundedVariance calculates a differentially private variance of a collection of
// float64 values.
//
// The output will be clamped between 0 and (upper - lower)² / 4. Since the result is
// guaranteed to be positive, this algorithm can be used to compute a differentially
// private standard deviation.
//
// The algorithm is a variation of the algorithm for differentially private mean
// from "Differential Privacy: From Theory to Practice", section 2.5.5:
// https://books.google.com/books?id=WFttDQAAQBAJ&pg=PA24#v=onepage&q&f=false
//
// BoundedVariance supports privacy units that contribute to multiple partitions
// (via the MaxPartitionsContributed parameter) as well as contribute to the same
// partition multiple times (via the MaxContributionsPerPartition parameter), by
// scaling the added noise appropriately.
//
// For general details and key definitions, see
// https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions.
//
// Note: Do not use when your results may cause overflows for float64 values. This
// aggregation is not hardened for such applications yet.
//
// Not thread-safe.
type BoundedVariance struct {
	// Parameters
	lower float64
	upper float64

	// State variables
	NormalizedSumOfSquares BoundedSumFloat64
	NormalizedSum          BoundedSumFloat64
	Count                  Count
	// The midpoint between lower and upper bounds. It cannot be set by the user;
	// it will be calculated based on the lower and upper values.
	midPoint float64
	state    aggregationState
}

func bvEquallyInitialized(bv1, bv2 *BoundedVariance) bool {
	return bv1.lower == bv2.lower &&
		bv1.upper == bv2.upper &&
		bv1.midPoint == bv2.midPoint &&
		bv1.state == bv2.state &&
		countEquallyInitialized(&bv1.Count, &bv2.Count) &&
		bsEquallyInitializedFloat64(&bv1.NormalizedSum, &bv2.NormalizedSum) &&
		bsEquallyInitializedFloat64(&bv1.NormalizedSumOfSquares, &bv2.NormalizedSumOfSquares)
}

// BoundedVarianceOptions contains the options necessary to initialize a BoundedVariance.
type BoundedVarianceOptions struct {
	Epsilon                      float64 // Privacy parameter ε. Required.
	Delta                        float64 // Privacy parameter δ. Required with Gaussian noise, must be 0 with Laplace noise.
	MaxPartitionsContributed     int64   // How many distinct partitions may a single user contribute to? Defaults to 1.
	MaxContributionsPerPartition int64   // How many times may a single user contribute to a single partition? Required.
	// Lower and Upper bounds for clamping. Default to 0; must be such that Lower < Upper.
	Lower, Upper float64
	Noise        noise.Noise // Type of noise used in BoundedVariance. Defaults to Laplace noise.
}

// NewBoundedVariance returns a new BoundedVariance.
func NewBoundedVariance(opt *BoundedVarianceOptions) *BoundedVariance {
	if opt == nil {
		opt = &BoundedVarianceOptions{}
	}

	maxContributionsPerPartition := opt.MaxContributionsPerPartition
	if maxContributionsPerPartition == 0 {
		// TODO: do not exit the program from within library code
		log.Fatalf("NewBoundedVariance requires a value for MaxContributionsPerPartition")
	}

	// Set defaults.
	maxPartitionsContributed := opt.MaxPartitionsContributed
	if maxPartitionsContributed == 0 {
		maxPartitionsContributed = 1
	}

	n := opt.Noise
	if n == nil {
		n = noise.Laplace()
	}
	// Check bounds & use them to compute L_∞ sensitivity.
	lower, upper := opt.Lower, opt.Upper
	if lower == 0 && upper == 0 {
		// TODO: do not exit the program from within library code
		log.Fatalf("NewBoundedVariance requires a non-default value for Lower or Upper(automatic bounds determination is not implemented yet). Lower and Upper cannot be both 0")
	}
	if err := checks.CheckBoundsFloat64("NewBoundedVariance", lower, upper); err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("CheckBoundsFloat64(lower %f, upper %f) failed with %v", lower, upper, err)
	}
	if err := checks.CheckBoundsNotEqual("NewBoundedVariance", lower, upper); err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("CheckBoundsNotEqual(lower %f, upper %f) failed with %v", lower, upper, err)
	}
	// (lower + upper) / 2 may cause an overflow if lower and upper are large values.
	midPoint := lower + (upper-lower)/2.0
	sumMaxDistFromMidpoint := upper - midPoint

	eps, del := opt.Epsilon, opt.Delta
	// We split the budget equally in three to calculate the count, the normalized sum and
	// normalized sum of squares.
	// TODO: This can be optimized.
	countEpsilon := eps / 3
	countDelta := del / 3
	sumEpsilon := eps / 3
	sumDelta := del / 3
	sumOfSquaresEpsilon := eps - countEpsilon - sumEpsilon
	sumOfSquaresDelta := del - countDelta - sumDelta

	// Check that the parameters are compatible with the noise chosen by calling
	// the noise on some dummy value.
	n.AddNoiseFloat64(0, 1, 1, countEpsilon, countDelta)
	n.AddNoiseFloat64(0, 1, 1, sumEpsilon, sumDelta)
	n.AddNoiseFloat64(0, 1, 1, sumOfSquaresEpsilon, sumOfSquaresDelta)

	// normalizedSumOfSquares s2 yields a differentially private sum of squares of the position of the
	// entries e_i relative to the midpoint m = (lower + upper) / 2 of the range of the bounded variance,
	// i.e., s2 = Σ_i (e_i - m) (e_i - m).
	//
	// normalizedSum s yields a differentially private sum of the position of the entries e_i relative
	// to the midpoint m = (lower + upper) / 2 of the range of the bounded variance, i.e., s = Σ_i (e_i - m).
	//
	// count c yields a differentially private count of the entries.
	//
	// Given normalized sum of squares s2, normalized sum s and count c (all without noise), the true variance
	// can be computed as (since variance is invariant to translation):
	// variance = s2 / c - (s / c)^2
	//
	// the rest follows from the code.
	count := NewCount(&CountOptions{
		Epsilon:                      countEpsilon,
		Delta:                        countDelta,
		MaxPartitionsContributed:     maxPartitionsContributed,
		Noise:                        n,
		maxContributionsPerPartition: maxContributionsPerPartition,
	})

	normalizedSum := NewBoundedSumFloat64(&BoundedSumFloat64Options{
		Epsilon:                      sumEpsilon,
		Delta:                        sumDelta,
		MaxPartitionsContributed:     maxPartitionsContributed,
		Lower:                        -sumMaxDistFromMidpoint,
		Upper:                        sumMaxDistFromMidpoint,
		Noise:                        n,
		maxContributionsPerPartition: maxContributionsPerPartition,
	})

	normalizedSumOfSquares := NewBoundedSumFloat64(&BoundedSumFloat64Options{
		Epsilon:                  sumOfSquaresEpsilon,
		Delta:                    sumOfSquaresDelta,
		MaxPartitionsContributed: maxPartitionsContributed,
		// TODO: Do a second round of normalization for halving the lInf by two.
		Lower:                        0,
		Upper:                        math.Pow(sumMaxDistFromMidpoint, 2),
		Noise:                        n,
		maxContributionsPerPartition: maxContributionsPerPartition,
	})

	return &BoundedVariance{
		lower:                  lower,
		upper:                  upper,
		midPoint:               midPoint,
		Count:                  *count,
		NormalizedSum:          *normalizedSum,
		NormalizedSumOfSquares: *normalizedSumOfSquares,
		state:                  defaultState,
	}
}

// computeMaxVariance returns the maximum possible variance that could result from
// given lower and upper bounds. Used as an upper bound to the noisy variance.
func computeMaxVariance(lower, upper float64) float64 {
	return (upper - lower) * (upper - lower) / 4
}

// Add an entry to a BoundedVariance. It skips NaN entries and doesn't count them in
// the final result because introducing even a single NaN entry will result in a NaN
// variance regardless of other entries, which would break the indistinguishability
// property required for differential privacy.
func (bv *BoundedVariance) Add(e float64) {
	if bv.state != defaultState {
		// TODO: do not exit the program from within library code
		log.Fatalf("Variance cannot be amended. Reason: %v", bv.state.errorMessage())
	}
	if !math.IsNaN(e) {
		clamped, err := ClampFloat64(e, bv.lower, bv.upper)
		if err != nil {
			// TODO: do not exit the program from within library code
			log.Fatalf("Couldn't clamp input value %v, err %v", e, err)
		}

		normalizedValSquared := math.Pow(clamped-bv.midPoint, 2)
		bv.NormalizedSumOfSquares.Add(normalizedValSquared)
		normalizedVal := clamped - bv.midPoint
		bv.NormalizedSum.Add(normalizedVal)
		bv.Count.Increment()
	}
}

// Result returns a differentially private estimate of the variance of bounded
// elements added so far. The method can be called only once.
//
// Note that the returned value is not an unbiased estimate of the raw bounded variance.
func (bv *BoundedVariance) Result() float64 {
	if bv.state != defaultState {
		// TODO: do not exit the program from within library code
		log.Fatalf("Variance's noised result cannot be computed. Reason: " + bv.state.errorMessage())
	}
	bv.state = resultReturned
	noisedCount := math.Max(1.0, float64(bv.Count.Result()))
	noisedSum := bv.NormalizedSum.Result()
	noisedSumOfSquares := bv.NormalizedSumOfSquares.Result()

	normalizedMean := noisedSum / noisedCount
	normalizedMeanOfSquares := noisedSumOfSquares / noisedCount

	clamped, err := ClampFloat64(normalizedMeanOfSquares-normalizedMean*normalizedMean, 0.0, computeMaxVariance(bv.lower, bv.upper))
	if err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("Couldn't clamp the result, err %v", err)
	}
	return clamped
}

// Merge merges bv2 into bv (i.e., adds to bv all entries that were added to
// bv2). bv2 is consumed by this operation: bv2 may not be used after it is
// merged into bv.
func (bv *BoundedVariance) Merge(bv2 *BoundedVariance) {
	if err := checkMergeBoundedVariance(bv, bv2); err != nil {
		// TODO: do not exit the program from within library code
		log.Exit(err)
	}
	bv.NormalizedSumOfSquares.Merge(&bv2.NormalizedSumOfSquares)
	bv.NormalizedSum.Merge(&bv2.NormalizedSum)
	bv.Count.Merge(&bv2.Count)
	bv2.state = merged
}

func checkMergeBoundedVariance(bv1, bv2 *BoundedVariance) error {
	if bv1.state != defaultState {
		return fmt.Errorf("checkMergeBoundedVariance: bv1 cannot be merged with another BoundedVariance instance. Reason: %v", bv1.state.errorMessage())
	}
	if bv2.state != defaultState {
		return fmt.Errorf("checkMergeBoundedVariance: bv2 cannot be merged with another BoundedVariance instance. Reason: %v", bv2.state.errorMessage())
	}

	if !bvEquallyInitialized(bv1, bv2) {
		return fmt.Errorf("checkMergeBoundedVariance: bv1 and bv2 are not compatible")
	}

	return nil
}

// GobEncode encodes BoundedVariance.
func (bv *BoundedVariance) GobEncode() ([]byte, error) {
	if bv.state != defaultState && bv.state != serialized {
		return nil, fmt.Errorf("Variance object cannot be serialized. Reason: " + bv.state.errorMessage())
	}
	enc := encodableBoundedVariance{
		Lower:                           bv.lower,
		Upper:                           bv.upper,
		EncodableCount:                  &bv.Count,
		EncodableNormalizedSum:          &bv.NormalizedSum,
		EncodableNormalizedSumOfSquares: &bv.NormalizedSumOfSquares,
		Midpoint:                        bv.midPoint,
	}
	bv.state = serialized
	return encode(enc)
}

// GobDecode decodes BoundedVariance.
func (bv *BoundedVariance) GobDecode(data []byte) error {
	var enc encodableBoundedVariance
	err := decode(&enc, data)
	if err != nil {
		log.Fatalf("GobDecode: couldn't decode BoundedVariance from bytes")
		return err
	}
	*bv = BoundedVariance{
		lower:                  enc.Lower,
		upper:                  enc.Upper,
		Count:                  *enc.EncodableCount,
		NormalizedSum:          *enc.EncodableNormalizedSum,
		NormalizedSumOfSquares: *enc.EncodableNormalizedSumOfSquares,
		midPoint:               enc.Midpoint,
		state:                  defaultState,
	}
	return nil
}

// encodableBoundedVariance can be encoded by the gob package.
type encodableBoundedVariance struct {
	Lower                           float64
	Upper                           float64
	EncodableCount                  *Count
	EncodableNormalizedSum          *BoundedSumFloat64
	EncodableNormalizedSumOfSquares *BoundedSumFloat64
	Midpoint                        float64
}
