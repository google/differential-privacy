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

// Package checks contains checks for differentially private functions.
package checks

import (
	"fmt"
	"math"

	log "github.com/golang/glog"
)

const (
	epsilonName = "Epsilon"
	deltaName   = "Delta"
)

func verifyName(defaultName string, nameSlice []string) (string, error) {
	var name string
	switch len(nameSlice) {
	case 0:
		name = defaultName
	case 1:
		name = nameSlice[0]
	default:
		// TODO Add instructions for filing bugs.
		return "", fmt.Errorf("This should never happen. There should be 0 or 1 'name' parameter, got %d", len(nameSlice))
	}
	return name, nil
}

// CheckEpsilonVeryStrict returns an error if ε is +∞ or less than 2⁻⁵⁰.
func CheckEpsilonVeryStrict(epsilon float64, name ...string) error {
	epsName, err := verifyName(epsilonName, name)
	if err != nil {
		return err
	}
	if epsilon < math.Exp2(-50.0) || math.IsInf(epsilon, 0) || math.IsNaN(epsilon) {
		return fmt.Errorf("%s is %f, must be at least 2^-50 and finite", epsName, epsilon)
	}
	return nil
}

// CheckEpsilonStrict returns an error if ε is nonpositive or +∞.
func CheckEpsilonStrict(epsilon float64, name ...string) error {
	epsName, err := verifyName(epsilonName, name)
	if err != nil {
		return err
	}
	if epsilon <= 0 || math.IsInf(epsilon, 0) || math.IsNaN(epsilon) {
		return fmt.Errorf("%s is %f, must be strictly positive and finite", epsName, epsilon)
	}
	return nil
}

// CheckEpsilon returns an error if ε is strictly negative or +∞.
func CheckEpsilon(epsilon float64, name ...string) error {
	epsName, err := verifyName(epsilonName, name)
	if err != nil {
		return err
	}
	if epsilon < 0 || math.IsInf(epsilon, 0) || math.IsNaN(epsilon) {
		return fmt.Errorf("%s is %f, must be nonnegative and finite", epsName, epsilon)
	}
	return nil
}

// CheckDelta returns an error if δ is negative or greater than or equal to 1.
func CheckDelta(delta float64, name ...string) error {
	delName, err := verifyName(deltaName, name)
	if err != nil {
		return err
	}
	if math.IsNaN(delta) {
		return fmt.Errorf("%s is %e, cannot be NaN", delName, delta)
	}
	if delta < 0 {
		return fmt.Errorf("%s is %e, cannot be negative", delName, delta)
	}
	if delta >= 1 {
		return fmt.Errorf("%s is %e, must be strictly less than 1", delName, delta)
	}
	return nil
}

// CheckDeltaStrict returns an error if δ is nonpositive or greater than or equal to 1.
func CheckDeltaStrict(delta float64, name ...string) error {
	delName, err := verifyName(deltaName, name)
	if err != nil {
		return err
	}
	if math.IsNaN(delta) {
		return fmt.Errorf("%s is %e, cannot be NaN", delName, delta)
	}
	if delta <= 0 {
		return fmt.Errorf("%s is %e, must be strictly positive", delName, delta)
	}
	if delta >= 1 {
		return fmt.Errorf("%s is %e, must be strictly less than 1", delName, delta)
	}
	return nil
}

// CheckNoDelta returns an error if δ is non-zero.
func CheckNoDelta(delta float64, name ...string) error {
	delName, err := verifyName(deltaName, name)
	if err != nil {
		return err
	}
	if delta != 0 {
		return fmt.Errorf("%s is %e, must be 0", delName, delta)
	}
	return nil
}

// CheckThresholdDelta returns an error if δ_threshold is nonpositive or greater than or
// equal to 1 or δ_threshold+δ_noise is greater than or equal to 1.
func CheckThresholdDelta(thresholdDelta, noiseDelta float64) error {
	if math.IsNaN(thresholdDelta) {
		return fmt.Errorf("ThresholdDelta is %e, cannot be NaN", thresholdDelta)
	}
	if thresholdDelta <= 0 {
		return fmt.Errorf("ThresholdDelta is %e, must be strictly positive", thresholdDelta)
	}
	if thresholdDelta >= 1 {
		return fmt.Errorf("ThresholdDelta is %e, must be strictly less than 1", thresholdDelta)
	}
	if thresholdDelta+noiseDelta >= 1 {
		return fmt.Errorf("ThresholdDelta+NoiseDelta is %e, must be strictly less than 1", thresholdDelta+noiseDelta)
	}
	return nil
}

// CheckL0Sensitivity returns an error if l0Sensitivity is nonpositive.
func CheckL0Sensitivity(l0Sensitivity int64) error {
	if l0Sensitivity <= 0 {
		return fmt.Errorf("L0Sensitivity is %d, must be strictly positive", l0Sensitivity)
	}
	return nil
}

// CheckLInfSensitivity returns an error if lInfSensitivity is nonpositive or +∞.
func CheckLInfSensitivity(lInfSensitivity float64) error {
	if lInfSensitivity <= 0 || math.IsInf(lInfSensitivity, 0) || math.IsNaN(lInfSensitivity) {
		return fmt.Errorf("LInfSensitivity is %f, must be strictly positive and finite", lInfSensitivity)
	}
	return nil
}

// CheckBoundsInt64 returns an error if lower is larger than upper, and ensures it won't lead to sensitivity overflow.
func CheckBoundsInt64(lower, upper int64) error {
	if lower == math.MinInt64 || upper == math.MinInt64 {
		return fmt.Errorf("Lower bound (%d) and upper bound (%d) must be strictly larger than MinInt64=%d to avoid sensitivity overflow", lower, upper, math.MinInt64)
	}
	if lower > upper {
		return fmt.Errorf("Upper bound (%d) must be larger than lower bound (%d)", upper, lower)
	}
	if lower == upper {
		log.Warningf("Lower bound is equal to upper bound: all added elements will be clamped to %d", upper)
	}
	return nil
}

// CheckBoundsInt64IgnoreOverflows returns an error if lower is larger than upper but ignores sensitivity overflows.
// This is used when noise is unrecognised.
func CheckBoundsInt64IgnoreOverflows(lower, upper int64) error {
	if lower > upper {
		return fmt.Errorf("Upper bound (%d) must be larger than lower bound (%d)", upper, lower)
	}
	if lower == upper {
		log.Warningf("Lower bound is equal to upper bound: all added elements will be clamped to %d", upper)
	}
	return nil
}

// CheckBoundsFloat64 returns an error if lower is larger than upper, or if either parameter is ±∞.
func CheckBoundsFloat64(lower, upper float64) error {
	if math.IsNaN(lower) {
		return fmt.Errorf("Lower bound cannot be NaN")
	}
	if math.IsNaN(upper) {
		return fmt.Errorf("Upper bound cannot be NaN")
	}
	if math.IsInf(lower, 0) {
		return fmt.Errorf("Lower bound cannot be infinity")
	}
	if math.IsInf(upper, 0) {
		return fmt.Errorf("Upper bound cannot be infinity")
	}
	if lower > upper {
		return fmt.Errorf("Upper bound (%f) must be larger than lower bound (%f)", upper, lower)
	}
	if lower == upper {
		log.Warningf("Lower bound is equal to upper bound: all added elements will be clamped to %f", upper)
	}
	return nil
}

// CheckBoundsFloat64IgnoreOverflows returns an error if lower is larger than upper but accepts either parameter being ±∞.
func CheckBoundsFloat64IgnoreOverflows(lower, upper float64) error {
	if math.IsNaN(lower) {
		return fmt.Errorf("Lower bound cannot be NaN")
	}
	if math.IsNaN(upper) {
		return fmt.Errorf("Upper bound cannot be NaN")
	}
	if lower > upper {
		return fmt.Errorf("Upper bound (%f) must be larger than lower bound(%f)", upper, lower)
	}
	if lower == upper {
		log.Warningf("Lower bound is equal to upper bound: all added elements will be clamped to %f", upper)
	}
	return nil
}

// CheckBoundsFloat64AsInt64 returns an error if lower is larger are NaN, or if either parameter overflow after conversion to int64.
func CheckBoundsFloat64AsInt64(lower, upper float64) error {
	if math.IsNaN(lower) {
		return fmt.Errorf("Lower bound cannot be NaN")
	}
	if math.IsNaN(upper) {
		return fmt.Errorf("Upper bound cannot be NaN")
	}
	maxInt := float64(math.MaxInt64)
	minInt := float64(math.MinInt64)
	if lower < minInt || lower > maxInt {
		return fmt.Errorf("Lower bound (%f) must be within [MinInt64=%f, MaxInt64=%f]", lower, minInt, maxInt)
	}
	if upper < minInt || upper > maxInt {
		return fmt.Errorf("Upper bound (%f) must be within [MinInt64=%f, MaxInt64=%f]", upper, minInt, maxInt)
	}
	return CheckBoundsInt64(int64(lower), int64(upper))
}

// CheckMaxContributionsPerPartition returns an error if maxContributionsPerPartition is nonpositive.
func CheckMaxContributionsPerPartition(maxContributionsPerPartition int64) error {
	if maxContributionsPerPartition <= 0 {
		return fmt.Errorf("MaxContributionsPerPartition (%d) must be set to a positive value", maxContributionsPerPartition)
	}
	return nil
}

// CheckAlpha returns an error if the supplied alpha is not between 0 and 1.
func CheckAlpha(alpha float64) error {
	if alpha <= 0 || alpha >= 1 || math.IsNaN(alpha) || math.IsInf(alpha, 0) {
		return fmt.Errorf("Alpha is %f, must be within (0, 1) and finite", alpha)
	}
	return nil
}

// CheckBoundsNotEqual returns an error if lower and upper bounds are equal.
func CheckBoundsNotEqual(lower, upper float64) error {
	if lower == upper {
		return fmt.Errorf("Lower and upper bounds are both %f, they cannot be equal to each other", lower)
	}
	return nil
}

// CheckTreeHeight returns an error if treeHeight is less than 1.
func CheckTreeHeight(treeHeight int) error {
	if treeHeight < 1 {
		return fmt.Errorf("Tree Height is %d, must be at least 1", treeHeight)
	}
	return nil
}

// CheckBranchingFactor returns an error if branchingFactor is less than 2.
func CheckBranchingFactor(branchingFactor int) error {
	if branchingFactor < 2 {
		return fmt.Errorf("Branching Factor is %d, must be at least 2", branchingFactor)
	}
	return nil
}

// CheckPreThreshold returns an error if preThreshold is less than 0.
func CheckPreThreshold(preThreshold int64) error {
	if preThreshold < 0 {
		return fmt.Errorf("PreThreshold is %d, must be at least 0", preThreshold)
	}
	return nil
}
