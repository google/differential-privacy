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

	"github.com/google/differential-privacy/go/v3/checks"
	"github.com/google/differential-privacy/go/v3/noise"
)

// Count calculates a differentially private count of a collection of values
// using the Laplace or Gaussian mechanism
//
// It supports privacy units that contribute to multiple partitions (via the
// MaxPartitionsContributed parameter) by scaling the added noise appropriately.
// However, it does not support multiple contributions to a single partition from
// the same privacy unit. For that use case, BoundedSumInt64 should be used instead.
//
// The provided differentially private count is an unbiased estimate of the raw
// count meaning that its expected value is equal to the raw count.
//
// For general details and key definitions, see
// https://github.com/google/differential-privacy/blob/main/differential_privacy.md#key-definitions.
//
// Note: Do not use when your results may cause overflows for int64 values.
// This aggregation is not hardened for such applications yet.
//
// Not thread-safe.
type Count struct {
	// Parameters
	epsilon         float64
	delta           float64
	l0Sensitivity   int64
	lInfSensitivity int64
	Noise           noise.Noise
	noiseKind       noise.Kind // necessary for serializing noise.Noise information

	// State variables
	count       int64
	state       aggregationState
	noisedCount int64
}

func countEquallyInitialized(c1, c2 *Count) bool {
	return c1.epsilon == c2.epsilon &&
		c1.delta == c2.delta &&
		c1.l0Sensitivity == c2.l0Sensitivity &&
		c1.lInfSensitivity == c2.lInfSensitivity &&
		c1.noiseKind == c2.noiseKind &&
		c1.state == c2.state
}

// CountOptions contains the options necessary to initialize a Count.
type CountOptions struct {
	Epsilon                  float64     // Privacy parameter ε. Required.
	Delta                    float64     // Privacy parameter δ. Required with Gaussian noise, must be 0 with Laplace noise.
	MaxPartitionsContributed int64       // How many distinct partitions may a single privacy unit contribute to? Required.
	Noise                    noise.Noise // Type of noise used. Defaults to Laplace noise.
	// How many times may a single privacy unit contribute to a single partition?
	// Defaults to 1. This is only needed for other aggregation functions using Count;
	// which is why the option is not exported.
	maxContributionsPerPartition int64
}

// NewCount returns a new Count, initialized at 0.
func NewCount(opt *CountOptions) (*Count, error) {
	if opt == nil {
		opt = &CountOptions{} // Prevents panicking due to a nil pointer dereference.
	}

	l0 := opt.MaxPartitionsContributed
	if l0 == 0 {
		return nil, fmt.Errorf("NewCount: MaxPartitionsContributed must be set")
	}

	lInf := opt.maxContributionsPerPartition
	if lInf == 0 {
		lInf = 1
	}

	n := opt.Noise
	if n == nil {
		n = noise.Laplace()
	}
	// Check that the parameters are compatible with the noise chosen by calling
	// the noise on some placeholder value.
	eps, del := opt.Epsilon, opt.Delta
	_, err := n.AddNoiseInt64(0, l0, lInf, eps, del)
	if err != nil {
		return nil, fmt.Errorf("NewCount: %w", err)
	}

	return &Count{
		epsilon:         eps,
		delta:           del,
		l0Sensitivity:   l0,
		lInfSensitivity: lInf,
		Noise:           n,
		noiseKind:       noise.ToKind(n),
		count:           0,
		state:           defaultState,
	}, nil
}

// Increment increments the count by one.
func (c *Count) Increment() error {
	return c.IncrementBy(1)
}

// IncrementBy increments the count by the given value.
// Note that this shouldn't be used to count multiple contributions to a
// single partition from the same privacy unit.
//
// It could, for example, be used to increment the count by k privacy
// units at once.
//
// Note that decrementing counts by inputting a negative value is allowed,
// for example if you want to remove some users you have previously added.
func (c *Count) IncrementBy(count int64) error {
	if c.state != defaultState {
		return fmt.Errorf("Count cannot be amended: %v", c.state.errorMessage())
	}
	c.count += count
	return nil
}

// Merge merges c2 into c (i.e., adds to c all entries that were added to c2).
// c2 is consumed by this operation: it may not be used after it is merged
// into c.
func (c *Count) Merge(c2 *Count) error {
	if e := checkMergeCount(c, c2); e != nil {
		return e
	}
	c.count += c2.count
	c2.state = merged
	return nil
}

func checkMergeCount(c1, c2 *Count) error {
	if c1.state != defaultState {
		return fmt.Errorf("checkMergeCount: c1 cannot be merged with another Count instance: %v", c1.state.errorMessage())
	}
	if c2.state != defaultState {
		return fmt.Errorf("checkMergeCount: c2 cannot be merged with another Count instance: %v", c2.state.errorMessage())
	}

	if !countEquallyInitialized(c1, c2) {
		return fmt.Errorf("checkMergeCount: c1 and c2 are not compatible")
	}

	return nil
}

// Result returns a differentially private estimate of the current count. The
// method can be called only once.
//
// The returned value is an unbiased estimate of the raw count.
//
// The returned value may sometimes be negative. This can be corrected by setting
// negative results to 0. Note that such post processing introduces bias to the
// result.
func (c *Count) Result() (int64, error) {
	if c.state != defaultState {
		return 0, fmt.Errorf("Count's noised result cannot be computed: %s", c.state.errorMessage())
	}
	c.state = resultReturned
	var err error
	c.noisedCount, err = c.Noise.AddNoiseInt64(c.count, c.l0Sensitivity, c.lInfSensitivity, c.epsilon, c.delta)
	return c.noisedCount, err
}

// ThresholdedResult is similar to Result() but applies thresholding to the result.
// So, if the result is less than the threshold specified by the parameters of Count
// as well as thresholdDelta, it returns nil. Otherwise, it returns the result.
//
// Note that partitions associated with nil results should not be published if the mere
// existence of partitions is determined by private data.
func (c *Count) ThresholdedResult(thresholdDelta float64) (*int64, error) {
	threshold, err := c.Noise.Threshold(c.l0Sensitivity, float64(c.lInfSensitivity), c.epsilon, c.delta, thresholdDelta)
	if err != nil {
		return nil, err
	}
	result, err := c.Result()
	if err != nil {
		return nil, err
	}
	// Rounding up the threshold when converting it to int64 to ensure that no DP guarantees
	// are violated due to a result being returned that is less than the fractional threshold.
	if result < int64(math.Ceil(threshold)) {
		return nil, nil
	}
	return &result, nil
}

// PreThresholdedResult is similar to ThresholdedResult() but applies a deterministic
// 'pre-threshold' before applying the differentially private threshold.
//
// So, if the raw count is less than the specified pre-threshold or if the noisy result
// is less than preThreshold+dpThreshold, it returns nil.
// Otherwise, it returns the result.
//
// Note that partitions associated with nil results should not be published if the mere
// existence of partitions is determined by private data.
func (c *Count) PreThresholdedResult(preThreshold int64, thresholdDelta float64) (*int64, error) {
	if err := checks.CheckPreThreshold(preThreshold); err != nil {
		return nil, fmt.Errorf("Count's Pre-Thresholded Result cannot be computed: %w", err)
	}
	// Set PreThreshold to default 1 if not specified.
	if preThreshold < 1 {
		preThreshold = 1
	}
	// Pre-thresholding guarantees that at least this number of unique contributions are in the
	// partition.
	if c.count < preThreshold {
		return nil, nil
	}

	threshold, err := c.Noise.Threshold(c.l0Sensitivity, float64(c.lInfSensitivity), c.epsilon, c.delta, thresholdDelta)
	if err != nil {
		return nil, err
	}
	result, err := c.Result()
	if err != nil {
		return nil, err
	}
	// Rounding up the threshold when converting it to int64 to ensure that no DP guarantees
	// are violated due to a result being returned that is less than the fractional threshold.
	if result < int64(math.Ceil(threshold))+(preThreshold-1) {
		// PreThreshold is set to 1 as the default, we subtract it here so it has no effect.
		// This subtraction also ensures that partition might be kept if preThreshold = count.
		return nil, nil
	}
	return &result, nil
}

// ComputeConfidenceInterval computes a confidence interval with integer bounds that
// contains the true count with a probability greater than or equal to 1 - alpha using
// the noised count computed by Result(). The computation is based exclusively on the
// noised count returned by Result(). Thus no privacy budget is consumed by this operation.
//
// Result() needs to be called before ComputeConfidenceInterval, otherwise this will return
// an error.
//
// See https://github.com/google/differential-privacy/tree/main/common_docs/confidence_intervals.md.
func (c *Count) ComputeConfidenceInterval(alpha float64) (noise.ConfidenceInterval, error) {
	if c.state != resultReturned {
		return noise.ConfidenceInterval{}, fmt.Errorf("Result() must be called before calling ComputeConfidenceInterval()")
	}
	confInt, err := c.Noise.ComputeConfidenceIntervalInt64(c.noisedCount, c.l0Sensitivity, c.lInfSensitivity, c.epsilon, c.delta, alpha)
	if err != nil {
		return noise.ConfidenceInterval{}, err
	}
	// True count cannot be negative.
	confInt.LowerBound, confInt.UpperBound = math.Max(0, confInt.LowerBound), math.Max(0, confInt.UpperBound)
	return confInt, nil
}

// encodableCount can be encoded by the gob package.
type encodableCount struct {
	Epsilon         float64
	Delta           float64
	L0Sensitivity   int64
	LInfSensitivity int64
	NoiseKind       noise.Kind
	Count           int64
}

// GobEncode encodes Count.
func (c *Count) GobEncode() ([]byte, error) {
	if c.state != defaultState && c.state != serialized {
		return nil, fmt.Errorf("Count object cannot be serialized: " + c.state.errorMessage())
	}
	enc := encodableCount{
		Epsilon:         c.epsilon,
		Delta:           c.delta,
		L0Sensitivity:   c.l0Sensitivity,
		LInfSensitivity: c.lInfSensitivity,
		NoiseKind:       noise.ToKind(c.Noise),
		Count:           c.count,
	}
	c.state = serialized
	return encode(enc)
}

// GobDecode decodes Count.
func (c *Count) GobDecode(data []byte) error {
	var enc encodableCount
	err := decode(&enc, data)
	if err != nil {
		return fmt.Errorf("couldn't decode Count from bytes")
	}
	*c = Count{
		epsilon:         enc.Epsilon,
		delta:           enc.Delta,
		l0Sensitivity:   enc.L0Sensitivity,
		lInfSensitivity: enc.LInfSensitivity,
		noiseKind:       enc.NoiseKind,
		Noise:           noise.ToNoise(enc.NoiseKind),
		count:           enc.Count,
		state:           defaultState,
	}
	return nil
}
