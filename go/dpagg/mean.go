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

package dpagg

import (
	"fmt"
	"math"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/noise"
)

// BoundedMeanFloat64 calculates a differentially private mean of a collection of
// float64 values.
//
// The mean is computed by dividing a noisy sum of the entries by a noisy count of
// the entries. To improve utility, all entries are normalized by setting them to
// the difference between their actual value and the middle of the input range
// before summation. The original mean is recovered by adding the midpoint in a
// post-processing step. This idea is taken from Algorithm 2.4 of "Differential
// Privacy: From Theory to Practice", by Ninghui Li, Min Lyu, Dong Su and Weining
// Yang (section 2.5.5, page 28). In contrast to Algorithm 2.4, we do not return
// the midpoint if the noisy count is less or equal to 1. Instead, we set the noisy
// count to 1. Since this is a mere post-processing step, the DP bounds are
// preserved. Moreover, for small numbers of entries, this approach will return
// results that are closer to the actual mean in expectation.
//
// BoundedMeanFloat64 supports privacy units that contribute to multiple partitions
// (via the MaxPartitionsContributed parameter) as well as contribute to the same
// partition multiple times (via the MaxContributionsPerPartition parameter), by
// scaling the added noise appropriately.
//
// For general details and key definitions, see
// https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions.
//
// Note: Do not use when your results may cause overflows for int64 or float64
// values. This aggregation is not hardened for such applications yet.
//
// Not thread-safe.
type BoundedMeanFloat64 struct {
	// Parameters
	lower float64
	upper float64

	// State variables
	normalizedSum BoundedSumFloat64
	count         Count
	// The midpoint between lower and upper bounds. It cannot be set by the user;
	// it will be calculated based on the lower and upper values.
	midPoint       float64
	resultReturned bool // whether the result has already been returned
}

func bmEquallyInitializedFloat64(bm1, bm2 *BoundedMeanFloat64) bool {
	return bm1.lower == bm2.lower &&
		bm1.upper == bm2.upper &&
		countEquallyInitialized(&bm1.count, &bm2.count) &&
		bsEquallyInitializedFloat64(&bm1.normalizedSum, &bm2.normalizedSum)
}

// BoundedMeanFloat64Options contains the options necessary to initialize a BoundedMeanFloat64.
type BoundedMeanFloat64Options struct {
	Epsilon                      float64     // Privacy parameter ε. Required.
	Delta                        float64     // Privacy parameter δ. Required with Gaussian noise, must be 0 with Laplace noise.
	MaxPartitionsContributed     int64       // How many distinct partitions may a single user contribute to? Defaults to 1.
	MaxContributionsPerPartition int64       // How many times may a single user contribute to a single partition? Required.
	// Lower and Upper bounds for clamping. Default to 0; must be such that Lower < Upper.
	Lower, Upper                 float64
	Noise                        noise.Noise // Type of noise used in BoundedMean. Defaults to Laplace noise.
}

// NewBoundedMeanFloat64 returns a new BoundedMeanFloat64.
func NewBoundedMeanFloat64(opt *BoundedMeanFloat64Options) *BoundedMeanFloat64 {
	if opt == nil {
		opt = &BoundedMeanFloat64Options{}
	}

	maxContributionsPerPartition := opt.MaxContributionsPerPartition
	if maxContributionsPerPartition == 0 {
		// TODO: do not exit the program from within library code
		log.Fatalf("NewBoundedMeanFloat64 requires a value for MaxContributionsPerPartition")
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
		log.Fatalf("NewBoundedMeanFloat64 requires a non-default value for Lower or Upper (automatic bounds determination is not implemented yet)")
	}
	if err := checks.CheckBoundsFloat64("NewBoundedMeanFloat64", lower, upper); err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("CheckBoundsFloat64(lower %f, upper %f) failed with %v", lower, upper, err)
	}
	// (lower + upper) / 2 may cause an overflow if lower and upper are large values.
	midPoint := lower + (upper-lower)/2.0
	maxDistFromMidpoint := math.Abs(upper - midPoint)

	eps, del := opt.Epsilon, opt.Delta
	// We split the budget in half to calculate the count and the noised normalized sum
	// TODO: this can be optimized for the Gaussian noise
	halfEpsilon := eps / 2
	halfDelta := del / 2

	// Check that the parameters are compatible with the noise chosen by calling
	// the noise on some dummy value.
	n.AddNoiseFloat64(0, 1, 1, halfEpsilon, halfDelta)

	// normalizedSum yields a differentially private sum of the position of the entries e_i relative
	// to the midpoint m = (lower + upper) / 2 of the range of the bounded mean, i.e., Σ_i (e_i - m)
	//
	// count yields a differentially private count of the entries.
	//
	// Given a normalized sum s and count c (both without noise), the true mean can be computed
	// as: mean =
	//   s / c + m =
	//   (Σ_i (e_i - m)) / c + m =
	//   (Σ_i (e_i - m)) / c + (Σ_i m) / c =
	//   (Σ_i e_i) / c
	//
	// the rest follows from the code.
	count := NewCount(&CountOptions{
		Epsilon:                      halfEpsilon,
		Delta:                        halfDelta,
		MaxPartitionsContributed:     maxPartitionsContributed,
		Noise:                        n,
		maxContributionsPerPartition: maxContributionsPerPartition,
	})

	normalizedSum := NewBoundedSumFloat64(&BoundedSumFloat64Options{
		Epsilon:                      halfEpsilon,
		Delta:                        halfDelta,
		MaxPartitionsContributed:     maxPartitionsContributed,
		Lower:                        -maxDistFromMidpoint,
		Upper:                        maxDistFromMidpoint,
		Noise:                        n,
		maxContributionsPerPartition: maxContributionsPerPartition,
	})

	return &BoundedMeanFloat64{
		lower:          lower,
		upper:          upper,
		midPoint:       midPoint,
		count:          *count,
		normalizedSum:  *normalizedSum,
		resultReturned: false,
	}
}

// Add an entry to a BoundedMeanFloat64. It skips NaN entries and doesn't count them in the final result
// because introducing even a single NaN entry will result in a NaN mean
// regardless of other entries, which would break the indistinguishability
// property required for differential privacy.
func (bm *BoundedMeanFloat64) Add(e float64) {
	if bm.resultReturned {
		// TODO: do not exit the program from within library code
		log.Fatalf("The mean has already been calculated and returned. It cannot be amended.")
	}
	if !math.IsNaN(e) {
		clamped, err := ClampFloat64(e, bm.lower, bm.upper)
		if err != nil {
			// TODO: do not exit the program from within library code
			log.Fatalf("Couldn't clamp input value %v, err %v", e, err)
		}

		x := clamped - bm.midPoint
		bm.normalizedSum.Add(x)
		bm.count.Increment()
	}
}

// Result returns a differentially private estimate of the average of bounded
// elements added so far. The method can be called only once.
//
// Note that the returned value is not an unbiased estimate of the raw bounded mean.
func (bm *BoundedMeanFloat64) Result() float64 {
	if bm.resultReturned {
		// TODO: do not exit the program from within library code
		log.Fatalf("The mean has already been calculated and returned. It can only be returned once.")
	}
	bm.resultReturned = true
	noisedCount := math.Max(1.0, float64(bm.count.Result()))
	noisedSum := bm.normalizedSum.Result()
	clamped, err := ClampFloat64(noisedSum/noisedCount+bm.midPoint, bm.lower, bm.upper)
	if err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("Couldn't clamp the result, err %v", err)
	}
	return clamped
}

// Merge merges bm2 into bm (i.e., adds to bm all entries that were added to
// bm2). bm2 is consumed by this operation: bm2 may not be used after it is
// merged into bm.
func (bm *BoundedMeanFloat64) Merge(bm2 *BoundedMeanFloat64) {
	if err := checkMergeBoundedMeanFloat64(bm, bm2); err != nil {
		// TODO: do not exit the program from within library code
		log.Exit(err)
	}
	bm.normalizedSum.sum += bm2.normalizedSum.sum
	bm.count.count += bm2.count.count
	bm2.resultReturned = true
}

func checkMergeBoundedMeanFloat64(bm1, bm2 *BoundedMeanFloat64) error {
	if bm1.resultReturned {
		return fmt.Errorf("checkMergeBoundedMeanFloat64: bm1 already returned the result, cannot be merged with another BoundedMean instance")
	}
	if bm2.resultReturned {
		return fmt.Errorf("checkMergeBoundedMeanFloat64: bm2 already returned the result, cannot be merged with another BoundedMean instance")
	}

	if !bmEquallyInitializedFloat64(bm1, bm2) {
		return fmt.Errorf("checkMergeBoundedMeanFloat64: bm1 and bm2 are not compatible")
	}

	return nil
}

// GobEncode encodes Count.
func (bm *BoundedMeanFloat64) GobEncode() ([]byte, error) {
	enc := encodableBoundedMeanFloat64{
		Lower:                  bm.lower,
		Upper:                  bm.upper,
		EncodableCount:         &bm.count,
		EncodableNormalizedSum: &bm.normalizedSum,
		MidPoint:               bm.midPoint,
		ResultReturned:         bm.resultReturned,
	}
	bm.resultReturned = true
	return encode(enc)
}

// GobDecode decodes Count.
func (bm *BoundedMeanFloat64) GobDecode(data []byte) error {
	var enc encodableBoundedMeanFloat64
	err := decode(&enc, data)
	if err != nil {
		log.Fatalf("GobDecode: couldn't decode BoundedMeanFloat64 from bytes")
		return err
	}
	*bm = BoundedMeanFloat64{
		lower:          enc.Lower,
		upper:          enc.Upper,
		count:          *enc.EncodableCount,
		normalizedSum:  *enc.EncodableNormalizedSum,
		midPoint:       enc.MidPoint,
		resultReturned: enc.ResultReturned,
	}
	return nil
}

// encodableBoundedMeanFloat64 can be encoded by the gob package.
type encodableBoundedMeanFloat64 struct {
	Lower                  float64
	Upper                  float64
	EncodableCount         *Count
	EncodableNormalizedSum *BoundedSumFloat64
	MidPoint               float64
	ResultReturned         bool
}
