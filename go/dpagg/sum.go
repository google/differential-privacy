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

// Package dpagg contains differentially private aggregation primitives.
package dpagg

import (
	"fmt"
	"math"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/noise"
)

// BoundedSumInt64 calculates a differentially private sum of a collection of
// int64 values. It supports privacy units that contribute to multiple partitions
// (via the MaxPartitionsContributed parameter) by scaling the added noise
// appropriately. However, it assumes that for each BoundedSumInt64 instance
// (partition), each privacy unit contributes at most one value. If a privacy unit
// contributes more, the contributions should be pre-aggregated before passing them
// to BoundedSumInt64.
//
// The provided differentially private sum is an unbiased estimate of the raw
// bounded sum in the sense that its expected value is equal to the raw bounded sum.
//
// For general details and key definitions, see
// https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions.
//
// Note: Do not use when your results may cause overflows for int64
// values. This aggregation is not hardened for such applications yet.
//
// Not thread-safe.
type BoundedSumInt64 struct {
	// Parameters
	epsilon         float64
	delta           float64
	l0Sensitivity   int64
	lInfSensitivity int64
	lower           int64
	upper           int64
	Noise           noise.Noise
	noiseKind       noise.Kind // necessary for serializing noise.Noise information

	// State variables
	sum       int64
	state     aggregationState
	noisedSum int64
}

func bsEquallyInitializedint64(s1, s2 *BoundedSumInt64) bool {
	return s1.epsilon == s2.epsilon &&
		s1.delta == s2.delta &&
		s1.l0Sensitivity == s2.l0Sensitivity &&
		s1.lInfSensitivity == s2.lInfSensitivity &&
		s1.lower == s2.lower &&
		s1.upper == s2.upper &&
		s1.noiseKind == s2.noiseKind &&
		s1.state == s2.state
}

// BoundedSumInt64Options contains the options necessary to initialize a BoundedSumInt64.
type BoundedSumInt64Options struct {
	Epsilon                  float64 // Privacy parameter ε. Required.
	Delta                    float64 // Privacy parameter δ. Required with Gaussian noise, must be 0 with Laplace noise.
	MaxPartitionsContributed int64   // How many distinct partitions may a single privacy unit contribute to? Defaults to 1.
	// Lower and Upper bounds for clamping. Default to 0; must be such that Lower < Upper.
	Lower, Upper int64
	Noise        noise.Noise // Type of noise used in BoundedSum. Defaults to Laplace noise.
	// How many times may a single privacy unit contribute to a single partition?
	// Defaults to 1. This is only needed for other aggregation functions using BoundedSum;
	// which is why the option is not exported.
	maxContributionsPerPartition int64
}

// NewBoundedSumInt64 returns a new BoundedSumInt64, whose sum is initialized at 0.
func NewBoundedSumInt64(opt *BoundedSumInt64Options) *BoundedSumInt64 {
	if opt == nil {
		opt = &BoundedSumInt64Options{}
	}
	// Set defaults.
	l0 := opt.MaxPartitionsContributed
	if l0 == 0 {
		l0 = 1
	}

	maxContributionsPerPartition := opt.maxContributionsPerPartition
	if maxContributionsPerPartition == 0 {
		maxContributionsPerPartition = 1
	}

	n := opt.Noise
	if n == nil {
		n = noise.Laplace()
	}
	// Check bounds & use them to compute L_∞ sensitivity
	lower, upper := opt.Lower, opt.Upper
	if lower == 0 && upper == 0 {
		// TODO: do not exit the program from within library code
		log.Fatalf("NewBoundedSumInt64 requires a non-default value for Lower or Upper (automatic bounds determination is not implemented yet). Lower and Upper cannot be both 0")
	}
	var err error
	switch noise.ToKind(opt.Noise) {
	case noise.Unrecognised:
		err = checks.CheckBoundsInt64IgnoreOverflows("NewBoundedSumInt64", lower, upper)
	default:
		err = checks.CheckBoundsInt64("NewBoundedSumInt64", lower, upper)
	}
	if err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("CheckBoundsInt64(lower %d, upper %d) failed with %v", lower, upper, err)
	}
	if err := checks.CheckBoundsNotEqual("NewBoundedSumInt64", float64(lower), float64(upper)); err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("CheckBoundsNotEqual(lower %f, upper %f) failed with %v", lower, upper, err)
	}
	lInf, err := getLInfInt(lower, upper, maxContributionsPerPartition)
	if err != nil {
		if noise.ToKind(opt.Noise) == noise.Unrecognised {
			// Ignore sensitivity overflows if noise is not recognised.
			log.Warningf("getLInfInt(lower %d, upper %d, maxContributionsPerPartition %d) failed with %v, using largest representable integer as lInf_sensitivity", lower, upper, maxContributionsPerPartition, err)
		} else {
			// TODO: do not exit the program from within library code
			log.Fatalf("getLInfInt(lower %d, upper %d, maxContributionsPerPartition %d) failed with %v", lower, upper, maxContributionsPerPartition, err)
		}
	}
	// Check that the parameters are compatible with the noise chosen by calling
	// the noise on some dummy value.
	eps, del := opt.Epsilon, opt.Delta
	n.AddNoiseInt64(0, l0, lInf, eps, del)

	return &BoundedSumInt64{
		epsilon:         eps,
		delta:           del,
		l0Sensitivity:   l0,
		lInfSensitivity: lInf,
		lower:           lower,
		upper:           upper,
		Noise:           n,
		noiseKind:       noise.ToKind(n),
		sum:             0,
		state:           defaultState,
	}
}

// lInfIntOverflows checks if multiplication of the given number overflows int64.
// If x != x*y/y then x*y overflowed and the multiplication result is incorrect.
// Thus, the equation evaluates to false.
func lInfIntOverflows(bound, maxContributionsPerPartition int64) bool {
	mult := bound * maxContributionsPerPartition
	return mult/maxContributionsPerPartition != bound
}

// getLInfInt checks that the sensitivity parameters will not create overflow errors,
// and returns the L_inf sensitivity of the BoundedSum object, which is calculated by the
// formula = max(|lower|, |upper|) * maxContributionsPerPartition.
func getLInfInt(lower, upper, maxContributionsPerPartition int64) (int64, error) {
	// If lower or upper is math.MinInt64, the sensitivity will overflow.
	if lower == math.MinInt64 || upper == math.MinInt64 {
		return math.MaxInt64, fmt.Errorf("lower = %d and upper = %d must be strictly larger than math.MinInt64 to avoid sensitivity overflow", lower, upper)
	}
	if lower < 0 {
		lower = -lower
	}
	if upper < 0 {
		upper = -upper
	}
	if lInfIntOverflows(lower, maxContributionsPerPartition) {
		return math.MaxInt64, fmt.Errorf(
			"lower = %d and maxContributionsPerPartition = %d are too high - the lInf sensitivity may overflow",
			lower, maxContributionsPerPartition)
	}
	if lInfIntOverflows(upper, maxContributionsPerPartition) {
		return math.MaxInt64, fmt.Errorf(
			"upper = %dr and maxContributionsPerPartition = %d are too high - the lInf sensitivity may overflow",
			upper, maxContributionsPerPartition)
	}
	if lower > upper {
		return lower * maxContributionsPerPartition, nil
	}
	return upper * maxContributionsPerPartition, nil
}

// Add adds a new summand to the BoundedSumInt64.
func (bs *BoundedSumInt64) Add(e int64) {
	if bs.state != defaultState {
		// TODO: do not exit the program from within library code
		log.Fatalf("Sum cannot be amended. Reason: %v", bs.state.errorMessage())
	}
	clamped, err := ClampInt64(e, bs.lower, bs.upper)
	if err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("Couldn't clamp input value %v, err %v", e, err)
	}
	bs.sum += clamped
}

// Merge merges bs2 into bs (i.e., adds to bs all entries that were added to
// bs2). bs2 is consumed by this operation: bs2 may not be used after it is
// merged into bs.
func (bs *BoundedSumInt64) Merge(bs2 *BoundedSumInt64) {
	if e := checkMergeBoundedSumInt64(bs, bs2); e != nil {
		log.Exit(e)
	}
	bs.sum += bs2.sum
	bs2.state = merged
}

func checkMergeBoundedSumInt64(bs1, bs2 *BoundedSumInt64) error {
	if bs1.state != defaultState {
		return fmt.Errorf("checkMergeBoundedSumInt64: bs1 cannot be merged with another BoundedSum instance. Reason: %v", bs1.state.errorMessage())
	}
	if bs2.state != defaultState {
		return fmt.Errorf("checkMergeBoundedSumInt64: bs2 cannot be merged with another BoundedSum instance. Reason: %v", bs2.state.errorMessage())
	}

	if !bsEquallyInitializedint64(bs1, bs2) {
		return fmt.Errorf("checkMergeBoundedSumInt64: bs1 and bs2 are not compatible")
	}
	return nil
}

// Result returns a differentially private estimate of the sum of bounded
// elements added so far. The method can be called only once.
//
// The returned value is an unbiased estimate of the raw bounded sum.
//
// The returned value may sometimes be outside the set of possible raw bounded
// sums, e.g., the differentially private bounded sum may be positive although
// neither the lower nor the upper bound are positive. This can be corrected
// by the caller of this method, e.g., by snapping the result to the closest
// value representing a bounded sum that is possible. Note that such post
// processing introduces bias to the result.
func (bs *BoundedSumInt64) Result() int64 {
	if bs.state != defaultState {
		// TODO: do not exit the program from within library code
		log.Fatalf("Sum's noised result cannot be computed. Reason: " + bs.state.errorMessage())
	}
	bs.state = resultReturned
	bs.noisedSum = bs.Noise.AddNoiseInt64(bs.sum, bs.l0Sensitivity, bs.lInfSensitivity, bs.epsilon, bs.delta)
	return bs.noisedSum
}

// ThresholdedResult is similar to Result() but applies thresholding to the
// result. So, if the result is less than the threshold specified by the noise
// mechanism, it returns nil. Otherwise, it returns the result.
func (bs *BoundedSumInt64) ThresholdedResult(thresholdDelta float64) *int64 {
	threshold := bs.Noise.Threshold(bs.l0Sensitivity, float64(bs.lInfSensitivity), bs.epsilon, bs.delta, thresholdDelta)
	result := bs.Result()
	// To make sure floating-point rounding doesn't break DP guarantees, we err on
	// the side of dropping the result if it is exactly equal to the threshold.
	if float64(result) <= threshold {
		return nil
	}
	return &result
}

// ComputeConfidenceInterval computes a confidence interval with integer bounds that
// contains the true sum with a probability greater than or equal to 1 - alpha using
// the noised sum computed by Result(). The computation is based exclusively on the
// noised sum returned by Result(). Thus no privacy budget is consumed by this operation.
//
// Result() needs to be called before ComputeConfidenceInterval, otherwise this will return
// an error.
//
// See https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md.
func (bs *BoundedSumInt64) ComputeConfidenceInterval(alpha float64) (noise.ConfidenceInterval, error) {
	if bs.state != resultReturned {
		return noise.ConfidenceInterval{}, fmt.Errorf("Result() must be called before calling ComputeConfidenceInterval()")
	}
	confInt, err := bs.Noise.ComputeConfidenceIntervalInt64(bs.noisedSum, bs.l0Sensitivity, bs.lInfSensitivity, bs.epsilon, bs.delta, alpha)
	if err != nil {
		return noise.ConfidenceInterval{}, err
	}
	// If lower and upper bounds are non-negative, trim the negative part of the interval.
	if bs.lower >= 0 {
		confInt.LowerBound, confInt.UpperBound = math.Max(0, confInt.LowerBound), math.Max(0, confInt.UpperBound)
	}
	// Similarly, if lower and upper bounds are non-positive, trim the positive part of the interval.
	if bs.upper <= 0 {
		confInt.LowerBound, confInt.UpperBound = math.Min(0, confInt.LowerBound), math.Min(0, confInt.UpperBound)
	}
	return confInt, nil
}

// encodableBoundedSumFloat64 can be encoded by the gob package.
type encodableBoundedSumInt64 struct {
	Epsilon         float64
	Delta           float64
	L0Sensitivity   int64
	LInfSensitivity int64
	Lower           int64
	Upper           int64
	NoiseKind       noise.Kind
	Sum             int64
}

// GobEncode encodes BoundedSumInt64.
func (bs *BoundedSumInt64) GobEncode() ([]byte, error) {
	if bs.state != defaultState {
		return nil, fmt.Errorf("Sum object cannot be serialized. Reason: " + bs.state.errorMessage())
	}
	enc := encodableBoundedSumInt64{
		Epsilon:         bs.epsilon,
		Delta:           bs.delta,
		L0Sensitivity:   bs.l0Sensitivity,
		LInfSensitivity: bs.lInfSensitivity,
		Lower:           bs.lower,
		Upper:           bs.upper,
		NoiseKind:       noise.ToKind(bs.Noise),
		Sum:             bs.sum,
	}
	bs.state = serialized
	return encode(enc)
}

// GobDecode decodes BoundedSumInt64.
func (bs *BoundedSumInt64) GobDecode(data []byte) error {
	var enc encodableBoundedSumInt64
	err := decode(&enc, data)
	if err != nil {
		log.Fatalf("GobDecode: couldn't decode BoundedSumInt64 from bytes")
		return err
	}
	*bs = BoundedSumInt64{
		epsilon:         enc.Epsilon,
		delta:           enc.Delta,
		l0Sensitivity:   enc.L0Sensitivity,
		lInfSensitivity: enc.LInfSensitivity,
		lower:           enc.Lower,
		upper:           enc.Upper,
		noiseKind:       enc.NoiseKind,
		Noise:           noise.ToNoise(enc.NoiseKind),
		sum:             enc.Sum,
		state:           defaultState,
	}
	return nil
}

// BoundedSumFloat64 calculates a differentially private sum of a collection of
// float64 values. It supports privacy units that contribute to multiple partitions
// (via the MaxPartitionsContributed parameter) by scaling the added noise
// appropriately. However, it assumes that for each BoundedSumFloat64 instance
// (partition), each privacy unit contributes at most one value. If a privacy unit
// contributes more, the contributions should be pre-aggregated before passing them
// to BoundedSumFloat64.
//
// The provided differentially private sum is an unbiased estimate of the raw
// bounded sum meaning that its expected value is equal to the raw bounded sum.
//
// For general details and key definitions, see
// https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions,
//
// Not thread-safe.
type BoundedSumFloat64 struct {
	// Parameters
	epsilon         float64
	delta           float64
	l0Sensitivity   int64
	lInfSensitivity float64
	lower           float64
	upper           float64
	Noise           noise.Noise
	noiseKind       noise.Kind // necessary for serializing noise.Noise information

	// State variables
	sum       float64
	state     aggregationState
	noisedSum float64
}

func bsEquallyInitializedFloat64(s1, s2 *BoundedSumFloat64) bool {
	return s1.epsilon == s2.epsilon &&
		s1.delta == s2.delta &&
		s1.l0Sensitivity == s2.l0Sensitivity &&
		s1.lInfSensitivity == s2.lInfSensitivity &&
		s1.lower == s2.lower &&
		s1.upper == s2.upper &&
		s1.noiseKind == s2.noiseKind &&
		s1.state == s2.state
}

// BoundedSumFloat64Options contains the options necessary to initialize a BoundedSumFloat64.
type BoundedSumFloat64Options struct {
	Epsilon                  float64 // Privacy parameter ε. Required.
	Delta                    float64 // Privacy parameter δ. Required with Gaussian noise, must be 0 with Laplace noise.
	MaxPartitionsContributed int64   // How many distinct partitions may a single privacy unit contribute to? Defaults to 1.
	// Lower and Upper bounds for clamping. Default to 0; must be such that Lower < Upper.
	Lower, Upper float64
	Noise        noise.Noise // Type of noise used in BoundedSum. Defaults to Laplace noise.
	// How many times may a single privacy unit contribute to a single partition?
	// Defaults to 1. This is only needed for other aggregation functions using BoundedSum;
	// which is why the option is not exported.
	maxContributionsPerPartition int64
}

// NewBoundedSumFloat64 returns a new BoundedSumFloat64, whose sum is initialized at 0.
func NewBoundedSumFloat64(opt *BoundedSumFloat64Options) *BoundedSumFloat64 {
	if opt == nil {
		opt = &BoundedSumFloat64Options{}
	}
	// Set defaults.
	l0 := opt.MaxPartitionsContributed
	if l0 == 0 {
		l0 = 1
	}

	maxContributionsPerPartition := opt.maxContributionsPerPartition
	if maxContributionsPerPartition == 0 {
		maxContributionsPerPartition = 1
	}

	n := opt.Noise
	if n == nil {
		n = noise.Laplace()
	}
	// Check bounds & use them to compute L_∞ sensitivity
	lower, upper := opt.Lower, opt.Upper
	if lower == 0 && upper == 0 {
		// TODO: do not exit the program from within library code
		log.Fatalf("NewBoundedSumFloat64 requires a non-default value for Lower or Upper (automatic bounds determination is not implemented yet). Lower and Upper cannot be both 0")
	}
	var err error
	switch noise.ToKind(opt.Noise) {
	case noise.Unrecognised:
		err = checks.CheckBoundsFloat64IgnoreOverflows("NewBoundedSumFloat64", lower, upper)
	default:
		err = checks.CheckBoundsFloat64("NewBoundedSumFloat64", lower, upper)
	}
	if err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("CheckBoundsFloat64(lower %f, upper %f) failed with %v", lower, upper, err)
	}
	if err := checks.CheckBoundsNotEqual("NewBoundedSumFloat64", lower, upper); err != nil {
		// TODO: do not exit the program from within library code
		log.Fatalf("CheckBoundsNotEqual(lower %f, upper %f) failed with %v", lower, upper, err)
	}
	lInf, err := getLInfFloat(lower, upper, maxContributionsPerPartition)
	if err != nil {
		if noise.ToKind(opt.Noise) == noise.Unrecognised {
			// Ignore sensitivity overflows if noise is not recognised.
			log.Warningf("getLInfFloat(lower %f, upper %f, maxContributionsPerPartition %d) failed with %v, using largest representable integer as lInf_sensitivity", lower, upper, maxContributionsPerPartition, err)
		} else {
			// TODO: do not exit the program from within library code
			log.Fatalf("getLInfFloat(lower %f, upper %f, maxContributionsPerPartition %d) failed with %v", lower, upper, maxContributionsPerPartition, err)
		}
	}
	// Check that the parameters are compatible with the noise chosen by calling
	// the noise on some dummy value.
	eps, del := opt.Epsilon, opt.Delta
	n.AddNoiseFloat64(0, l0, lInf, eps, del)

	return &BoundedSumFloat64{
		epsilon:         eps,
		delta:           del,
		l0Sensitivity:   l0,
		lInfSensitivity: lInf,
		lower:           lower,
		upper:           upper,
		Noise:           n,
		noiseKind:       noise.ToKind(n),
		sum:             0,
		state:           defaultState,
	}
}

func lInfFloatOverflows(bound float64, maxContributionsPerPartition int64) bool {
	return math.IsInf(bound*float64(maxContributionsPerPartition), 0)
}

// getLInfFloat checks that the sensitivity parameters will not create overflow errors,
// and returns the L_inf sensitivity of the BoundedSum object, which is calculated by the
// formula = max(|lower|, |upper|) * maxContributionsPerPartition.
func getLInfFloat(lower, upper float64, maxContributionsPerPartition int64) (float64, error) {
	if lower < 0 {
		lower = -lower
	}
	if upper < 0 {
		upper = -upper
	}
	if lInfFloatOverflows(lower, maxContributionsPerPartition) {
		return math.Inf(1), fmt.Errorf(
			"lower = %f and maxContributionsPerPartition =%d are too high - the lInf sensitivity may overflow",
			lower, maxContributionsPerPartition)
	}
	if lInfFloatOverflows(upper, maxContributionsPerPartition) {
		return math.Inf(1), fmt.Errorf(
			"upper = %f and maxContributionsPerPartition = %d are too high - the lInf sensitivity may overflow",
			upper, maxContributionsPerPartition)
	}
	if lower > upper {
		return lower * float64(maxContributionsPerPartition), nil
	}
	return upper * float64(maxContributionsPerPartition), nil
}

// Add adds a new summand to the BoundedSumFloat64. It ignores NaN summands
// because introducing even a single NaN summand will result in a NaN sum
// regardless of other summands, which would break the indistinguishability
// property required for differential privacy.
func (bs *BoundedSumFloat64) Add(e float64) {
	if bs.state != defaultState {
		// TODO: do not exit the program from within library code
		log.Fatalf("Sum cannot be amended. Reason: %v", bs.state.errorMessage())
	}
	if !math.IsNaN(e) {
		clamped, err := ClampFloat64(e, bs.lower, bs.upper)
		if err != nil {
			// TODO: do not exit the program from within library code
			log.Fatalf("Couldn't clamp input value %v, err %v", e, err)
		}
		bs.sum += clamped
	}
}

// Merge merges bs2 into bs (i.e., adds to bs all entries that were added to
// bs2). bs2 is consumed by this operation: bs2 may not be used after it is
// merged into bs.
func (bs *BoundedSumFloat64) Merge(bs2 *BoundedSumFloat64) {
	if e := checkMergeBoundedSumFloat64(bs, bs2); e != nil {
		log.Exit(e)
	}
	bs.sum += bs2.sum
	bs2.state = merged
}

func checkMergeBoundedSumFloat64(bs1, bs2 *BoundedSumFloat64) error {
	if bs1.state != defaultState {
		return fmt.Errorf("checkMergeBoundedSumFloat64: bs1 cannot be merged with another BoundedSum instance. Reason: %v", bs1.state.errorMessage())
	}
	if bs2.state != defaultState {
		return fmt.Errorf("checkMergeBoundedSumFloat64: bs2 cannot be merged with another BoundedSum instance. Reason: %v", bs2.state.errorMessage())
	}

	if !bsEquallyInitializedFloat64(bs1, bs2) {
		return fmt.Errorf("checkMergeBoundedSumFloat64: bs1 and bs2 are not compatible")
	}
	return nil
}

// Result returns a differentially private estimate of the sum of bounded
// elements added so far. The method can be called only once.
//
// The returned value is an unbiased estimate of the raw bounded sum.
//
// The returned value may sometimes be outside the set of possible raw bounded
// sums, e.g., the differentially private bounded sum may be positive although
// neither the lower nor the upper bound are positive. This can be corrected
// by the caller of this method, e.g., by snapping the result to the closest
// value representing a bounded sum that is possible. Note that such post
// processing introduces bias to the result.
func (bs *BoundedSumFloat64) Result() float64 {
	if bs.state != defaultState {
		// TODO: do not exit the program from within library code
		log.Fatalf("Sum's noised result cannot be computed. Reason: " + bs.state.errorMessage())
	}
	bs.state = resultReturned
	bs.noisedSum = bs.Noise.AddNoiseFloat64(bs.sum, bs.l0Sensitivity, bs.lInfSensitivity, bs.epsilon, bs.delta)
	return bs.noisedSum
}

// ThresholdedResult is similar to Result() but applies thresholding to the
// result. So, if the result is less than the threshold specified by the noise,
// mechanism, it returns nil. Otherwise, it returns the result.
func (bs *BoundedSumFloat64) ThresholdedResult(thresholdDela float64) *float64 {
	threshold := bs.Noise.Threshold(bs.l0Sensitivity, bs.lInfSensitivity, bs.epsilon, bs.delta, thresholdDela)
	result := bs.Result()
	if result < threshold {
		return nil
	}
	return &result
}

// ComputeConfidenceInterval computes a confidence interval that contains the true sum
// with a probability greater than or equal to 1 - alpha using the noised sum computed by
// Result(). The computation is based exclusively on the noised sum returned by Result().
// Thus no privacy budget is consumed by this operation.
//
// Result() needs to be called before ComputeConfidenceInterval, otherwise this will return
// an error.
//
// See https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md.
func (bs *BoundedSumFloat64) ComputeConfidenceInterval(alpha float64) (noise.ConfidenceInterval, error) {
	if bs.state != resultReturned {
		return noise.ConfidenceInterval{}, fmt.Errorf("Result() must be called before calling ComputeConfidenceInterval()")
	}
	confInt, err := bs.Noise.ComputeConfidenceIntervalFloat64(bs.noisedSum, bs.l0Sensitivity, bs.lInfSensitivity, bs.epsilon, bs.delta, alpha)
	if err != nil {
		return noise.ConfidenceInterval{}, err
	}
	// If lower and upper bounds are non-negative, trim the negative part of the interval.
	if bs.lower >= 0 {
		confInt.LowerBound, confInt.UpperBound = math.Max(0, confInt.LowerBound), math.Max(0, confInt.UpperBound)
	}
	// Similarly if lower and upper bounds are non-positive, trim the positive part of the interval.
	if bs.upper <= 0 {
		confInt.LowerBound, confInt.UpperBound = math.Min(0, confInt.LowerBound), math.Min(0, confInt.UpperBound)
	}
	return confInt, nil
}

// encodableBoundedSumFloat64 can be encoded by the gob package.
type encodableBoundedSumFloat64 struct {
	Epsilon         float64
	Delta           float64
	L0Sensitivity   int64
	LInfSensitivity float64
	Lower           float64
	Upper           float64
	NoiseKind       noise.Kind
	Sum             float64
}

// GobEncode encodes BoundedSumInt64.
func (bs *BoundedSumFloat64) GobEncode() ([]byte, error) {
	if bs.state != defaultState {
		return nil, fmt.Errorf("Sum object cannot be serialized. Reason: " + bs.state.errorMessage())
	}
	enc := encodableBoundedSumFloat64{
		Epsilon:         bs.epsilon,
		Delta:           bs.delta,
		L0Sensitivity:   bs.l0Sensitivity,
		LInfSensitivity: bs.lInfSensitivity,
		Lower:           bs.lower,
		Upper:           bs.upper,
		NoiseKind:       noise.ToKind(bs.Noise),
		Sum:             bs.sum,
	}
	bs.state = serialized
	return encode(enc)
}

// GobDecode decodes BoundedSumInt64.
func (bs *BoundedSumFloat64) GobDecode(data []byte) error {
	var enc encodableBoundedSumFloat64
	err := decode(&enc, data)
	if err != nil {
		log.Fatalf("GobDecode: couldn't decode BoundedSumFloat64 from bytes")
		return err
	}
	*bs = BoundedSumFloat64{
		epsilon:         enc.Epsilon,
		delta:           enc.Delta,
		l0Sensitivity:   enc.L0Sensitivity,
		lInfSensitivity: enc.LInfSensitivity,
		lower:           enc.Lower,
		upper:           enc.Upper,
		noiseKind:       enc.NoiseKind,
		Noise:           noise.ToNoise(enc.NoiseKind),
		sum:             enc.Sum,
		state:           defaultState,
	}
	return nil
}
