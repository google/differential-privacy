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

package rand

import (
	"bufio"
	"bytes"
	cryptorand "crypto/rand"
	"encoding/binary"
	"math"
	"sync"
	"testing"

	"github.com/google/differential-privacy/go/v4/stattestutils"
)

const (
	rangeSamples = 1e4 // Range/contract checks.
	statSamples  = 1e6 // Distributional moment checks.
)

// setUp restores the package-level RNG state at the end of the test.
// Tests in this file mutate that state, so they cannot use t.Parallel.
func setUp(t *testing.T) {
	t.Helper()
	origBuf := randBuf
	origBitBuf := randBitBuf
	origBitPos := randBitPos
	t.Cleanup(func() {
		randBuf = origBuf
		randBitBuf = origBitBuf
		randBitPos = origBitPos
	})
}

// withBytes feeds b as the source of randomness for the current test.
func withBytes(t *testing.T, b []byte) {
	t.Helper()
	setUp(t)
	randBuf = bytes.NewReader(b)
	randBitBuf = 0
	randBitPos = math.MaxInt8
}

// skipShort skips a long-running statistical test under -short.
func skipShort(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("statistical test")
	}
}

// withCryptoRand resets the randomness source to crypto/rand.
func withCryptoRand(t *testing.T) {
	t.Helper()
	setUp(t)
	randBuf = bufio.NewReaderSize(cryptorand.Reader, 65536)
	randBitBuf = 0
	randBitPos = math.MaxInt8
}

func TestU64_DecodesLittleEndian(t *testing.T) {
	for _, tc := range []struct {
		name  string
		bytes []byte
		want  uint64
	}{
		{"sequential_bytes", []byte{0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08}, 0x0807060504030201},
		{"all_zero", []byte{0, 0, 0, 0, 0, 0, 0, 0}, 0},
		{"all_one", []byte{0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, math.MaxUint64},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withBytes(t, tc.bytes)
			if got := U64(); got != tc.want {
				t.Errorf("U64() = %#x, want %#x", got, tc.want)
			}
		})
	}
}

func TestU8_ReadsSingleByte(t *testing.T) {
	withBytes(t, []byte{0xAB, 0xCD})
	if got, want := U8(), uint8(0xAB); got != want {
		t.Errorf("U8() first call = %#x, want %#x", got, want)
	}
	if got, want := U8(), uint8(0xCD); got != want {
		t.Errorf("U8() second call = %#x, want %#x", got, want)
	}
}

func TestBooleanBufIsShifting(t *testing.T) {
	withBytes(t, []byte{
		0b00100100,
		0b10010000,
	})
	for pos, want := range []bool{
		// first byte
		false,
		false,
		true,
		false,
		false,
		true,
		false,
		false,
		// second byte
		false,
		false,
		false,
		false,
		true,
		false,
		false,
		true,
	} {
		if got := Boolean(); got != want {
			t.Errorf("Boolean: got %v, want %v in %v-th iteration", got, want, pos)
		}
	}
}

func TestSign_FollowsBooleanLSBFirst(t *testing.T) {
	// Bit 0 is set, bit 1 is not, so Sign returns +1 then -1.
	withBytes(t, []byte{0b00000001})
	if got, want := Sign(), 1.0; got != want {
		t.Errorf("Sign() first call = %v, want %v", got, want)
	}
	if got, want := Sign(), -1.0; got != want {
		t.Errorf("Sign() second call = %v, want %v", got, want)
	}
}

func TestGeometric_CountsLeadingZeros(t *testing.T) {
	// want = 1 + total leading-zero bit count up to and including the
	// run of zero bytes preceding the first 1 bit. Each fully-zero byte
	// contributes 8.
	for _, tc := range []struct {
		name  string
		bytes []byte
		want  float64
	}{
		{name: "high_bit_set_no_leading_zeros", bytes: []byte{0x80}, want: 1.0},
		{name: "low_bit_set_seven_leading_zeros", bytes: []byte{0x01}, want: 8.0},
		{name: "two_zero_bytes_then_low_bit", bytes: []byte{0x00, 0x00, 0x01}, want: 24.0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withBytes(t, tc.bytes)
			if got := Geometric(); got != tc.want {
				t.Errorf("Geometric() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestI63n_RejectionLoopAtBoundary(t *testing.T) {
	// For n=3, largestMultipleOfN = (MaxInt64 / 3) * 3 = MaxInt64 - 1.
	// The first U64 lands exactly on the boundary and is rejected; the
	// second is one below the boundary and is accepted.
	const n int64 = 3
	largestMultipleOfN := (int64(math.MaxInt64) / n) * n
	buf := make([]byte, 16)
	binary.LittleEndian.PutUint64(buf[0:8], uint64(largestMultipleOfN))
	binary.LittleEndian.PutUint64(buf[8:16], uint64(largestMultipleOfN-1))
	withBytes(t, buf)

	want := (largestMultipleOfN - 1) % n
	if got := I63n(n); got != want {
		t.Errorf("I63n(%d) = %d, want %d", n, got, want)
	}
}

func TestI63n_MasksSignBit(t *testing.T) {
	// Without masking, int64(0x8000000000000005) is negative and Go's %
	// would return a negative remainder, violating [0, n).
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, 0x8000000000000005)
	withBytes(t, buf)
	if got, want := I63n(3), int64(2); got != want {
		t.Errorf("I63n(3) with sign bit set = %d, want %d", got, want)
	}
}

func TestUniform_ReturnsOneOnUnderflow(t *testing.T) {
	// Force Geometric() >= 1024 so 2^k overflows to +Inf and Uniform falls
	// back to 1.0. 8 bytes for U64, 128 zeros give 1024 leading bits, 0x80
	// terminates.
	buf := make([]byte, 8+128+1)
	buf[len(buf)-1] = 0x80
	withBytes(t, buf)
	if got := Uniform(); got != 1.0 {
		t.Errorf("Uniform() with overflowing Geometric = %v, want 1.0", got)
	}
}

func TestSign_RangeContract(t *testing.T) {
	withCryptoRand(t)
	for i := 0; i < rangeSamples; i++ {
		v := Sign()
		if v != 1.0 && v != -1.0 {
			t.Fatalf("Sign() returned %v at iteration %d, want -1 or +1", v, i)
		}
	}
}

func TestI63n_RangeContract(t *testing.T) {
	withCryptoRand(t)
	for _, n := range []int64{1, 2, 7, 1024, math.MaxInt64} {
		for i := 0; i < rangeSamples; i++ {
			v := I63n(n)
			if v < 0 || v >= n {
				t.Fatalf("I63n(%d) returned %d at iteration %d, want in [0, %d)", n, v, i, n)
			}
		}
	}
}

func TestUniform_RangeContract(t *testing.T) {
	withCryptoRand(t)
	for i := 0; i < rangeSamples; i++ {
		v := Uniform()
		if math.IsNaN(v) || v <= 0 || v > 1 {
			t.Fatalf("Uniform() returned %v at iteration %d, want in (0, 1]", v, i)
		}
	}
}

func TestGeometric_RangeContract(t *testing.T) {
	withCryptoRand(t)
	for i := 0; i < rangeSamples; i++ {
		v := Geometric()
		if v < 1 || v != math.Trunc(v) {
			t.Fatalf("Geometric() returned %v at iteration %d, want a positive integer", v, i)
		}
	}
}

func TestNormal_FiniteContract(t *testing.T) {
	withCryptoRand(t)
	for i := 0; i < rangeSamples; i++ {
		v := Normal()
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("Normal() returned %v at iteration %d, want finite", v, i)
		}
	}
}

func TestBoolean_MeanIsHalf(t *testing.T) {
	// The sample variance of a Bernoulli is functionally determined by
	// the mean, so a variance check would be degenerate.
	skipShort(t)
	withCryptoRand(t)
	trueCount := 0
	for i := 0; i < statSamples; i++ {
		if Boolean() {
			trueCount++
		}
	}
	mean := float64(trueCount) / float64(statSamples)
	// 5 standard deviations of the sample mean: 5 * sqrt(0.25/N).
	tol := 5 * math.Sqrt(0.25/float64(statSamples))
	if math.Abs(mean-0.5) > tol {
		t.Errorf("Boolean true fraction = %v, want 0.5 +/- %v", mean, tol)
	}
}

func TestSign_MeanAndVariance(t *testing.T) {
	skipShort(t)
	withCryptoRand(t)
	samples := make([]float64, statSamples)
	for i := range samples {
		samples[i] = Sign()
	}
	// {-1,+1} symmetric Bernoulli: mean 0, variance 1, mu_4 = 1.
	assertMomentsWithinFiveSigma(t, samples, 0.0, 1.0, 1.0)
}

func TestI63n_UniformMoments(t *testing.T) {
	// Discrete uniform on {0, ..., n-1}: mean (n-1)/2, variance (n^2-1)/12,
	// fourth central moment (n^2-1)(3n^2-7)/240. n capped at 2^20 to keep
	// the variance sum exact in float64.
	skipShort(t)
	for _, tc := range []struct {
		name string
		n    int64
	}{
		{"n=2", 2},
		{"n=7", 7},
		{"n=1024", 1024},
		{"n=1048576", 1 << 20},
	} {
		t.Run(tc.name, func(t *testing.T) {
			withCryptoRand(t)
			samples := make([]float64, statSamples)
			for i := range samples {
				samples[i] = float64(I63n(tc.n))
			}
			n2 := float64(tc.n) * float64(tc.n)
			wantMean := float64(tc.n-1) / 2.0
			wantVar := (n2 - 1) / 12.0
			muFour := (n2 - 1) * (3*n2 - 7) / 240.0
			assertMomentsWithinFiveSigma(t, samples, wantMean, wantVar, muFour)
		})
	}
}

func TestUniform_MeanAndVariance(t *testing.T) {
	skipShort(t)
	withCryptoRand(t)
	samples := make([]float64, statSamples)
	for i := range samples {
		samples[i] = Uniform()
	}
	// Continuous uniform on (0,1]: mean 1/2, variance 1/12,
	// fourth central moment 1/80.
	assertMomentsWithinFiveSigma(t, samples, 0.5, 1.0/12.0, 1.0/80.0)
}

func TestGeometric_MeanAndVariance(t *testing.T) {
	skipShort(t)
	withCryptoRand(t)
	samples := make([]float64, statSamples)
	for i := range samples {
		samples[i] = Geometric()
	}
	// Geometric(p=1/2) on {1,2,...}: mean 1/p = 2, variance (1-p)/p^2 = 2.
	assertMomentsWithinFiveSigma(t, samples, 2.0, 2.0, 38.0)
}

func TestNormal_MeanAndVariance(t *testing.T) {
	skipShort(t)
	withCryptoRand(t)
	samples := make([]float64, statSamples)
	for i := range samples {
		samples[i] = Normal()
	}
	// Standard normal: mean 0, variance 1, fourth central moment 3.
	assertMomentsWithinFiveSigma(t, samples, 0.0, 1.0, 3.0)
}

func TestNormal_FourthMoment(t *testing.T) {
	// Catches a Normal/Laplace swap: same mean and variance but mu_4
	// differs (3 vs 6).
	skipShort(t)
	withCryptoRand(t)
	samples := make([]float64, statSamples)
	for i := range samples {
		samples[i] = Normal()
	}
	mean := stattestutils.SampleMean(samples)
	var m4 float64
	for _, x := range samples {
		d := x - mean
		m4 += d * d * d * d
	}
	m4 /= float64(statSamples)
	tol := 5 * math.Sqrt((105.0-9.0)/float64(statSamples))
	if math.Abs(m4-3.0) > tol {
		t.Errorf("sample fourth central moment = %v, want 3.0 +/- %v (Laplace would give 6.0)", m4, tol)
	}
}

// TestConcurrent stresses the package locks. The race detector flags
// memory races; the aggregate Boolean mean check then catches lock
// failures that bias the marginal distribution.
func TestConcurrent(t *testing.T) {
	withCryptoRand(t)
	const goroutines = 8
	const callsPerGoroutine = 1000
	total := goroutines * callsPerGoroutine

	booleans := make([]bool, total)
	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			base := g * callsPerGoroutine
			for j := 0; j < callsPerGoroutine; j++ {
				_ = U64()
				_ = U8()
				booleans[base+j] = Boolean()
				_ = Sign()
				_ = I63n(100)
				_ = Uniform()
				_ = Geometric()
				_ = Normal()
			}
		}(g)
	}
	wg.Wait()

	trueCount := 0
	for _, b := range booleans {
		if b {
			trueCount++
		}
	}
	mean := float64(trueCount) / float64(total)
	tol := 5 * math.Sqrt(0.25/float64(total))
	if math.Abs(mean-0.5) > tol {
		t.Errorf("Boolean true fraction under concurrent access = %v, want 0.5 +/- %v", mean, tol)
	}
}

// 5-sigma check on the sample mean and variance. Skips the variance
// check when mu_4 == sigma^4.
func assertMomentsWithinFiveSigma(t *testing.T, samples []float64, wantMean, wantVar, muFour float64) {
	t.Helper()
	n := float64(len(samples))

	gotMean := stattestutils.SampleMean(samples)
	tol := 5 * math.Sqrt(wantVar/n)
	if math.Abs(gotMean-wantMean) > tol {
		t.Errorf("sample mean = %v, want %v +/- %v", gotMean, wantMean, tol)
	}

	varSESquared := (muFour - wantVar*wantVar) / n
	if varSESquared <= 0 {
		return
	}
	gotVar := stattestutils.SampleVariance(samples)
	tol = 5 * math.Sqrt(varSESquared)
	if math.Abs(gotVar-wantVar) > tol {
		t.Errorf("sample variance = %v, want %v +/- %v", gotVar, wantVar, tol)
	}
}
