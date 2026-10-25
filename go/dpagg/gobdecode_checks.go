package dpagg

import (
	"fmt"

	"github.com/google/differential-privacy/go/v4/checks"
	"github.com/google/differential-privacy/go/v4/noise"
)

// validateDecodedNoise rejects NoiseKind values that don't correspond to a
// concrete Noise implementation; without this gate, Result() panics on a nil
// receiver after deserializing a tampered payload.
func validateDecodedNoise(kind noise.Kind) (noise.Noise, error) {
	n := noise.ToNoise(kind)
	if n == nil {
		return nil, fmt.Errorf("unsupported NoiseKind value %d", kind)
	}
	return n, nil
}

// validateDecodedAggregation mirrors the precondition checks that New*()
// constructors already perform, applied to fields recovered from gob bytes.
func validateDecodedAggregation(epsilon, delta, lInfSensitivity float64, l0Sensitivity int64, kind noise.Kind) (noise.Noise, error) {
	n, err := validateDecodedNoise(kind)
	if err != nil {
		return nil, err
	}
	if err := checks.CheckEpsilonVeryStrict(epsilon); err != nil {
		return nil, err
	}
	if err := checks.CheckDelta(delta); err != nil {
		return nil, err
	}
	if err := checks.CheckL0Sensitivity(l0Sensitivity); err != nil {
		return nil, err
	}
	if err := checks.CheckLInfSensitivity(lInfSensitivity); err != nil {
		return nil, err
	}
	return n, nil
}
