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

// Package rand provides methods for generating random numbers from
// distributions useful for the differential privacy library.
package rand

import (
	"bufio"
	cryptorand "crypto/rand"
	"encoding/binary"
	"io"
	"math"
	"math/bits"
	mathrand "math/rand"
	"sync"

	log "github.com/golang/glog"
)

// TODO: Add test coverage for the various exported
// noise-generating functions.

var (
	randBufLock sync.Mutex
	randBuf     io.Reader = bufio.NewReaderSize(cryptorand.Reader, 65536)

	randBitLock sync.Mutex
	randBitBuf  uint8
	randBitPos  int8 = math.MaxInt8
)

func readRandBuf(b []byte) (int, error) {
	randBufLock.Lock()
	defer randBufLock.Unlock()
	return io.ReadFull(randBuf, b)
}

// U64 returns a uniformly random uint64.
func U64() uint64 {
	var r [8]uint8
	if _, err := readRandBuf(r[:]); err != nil {
		log.Fatalf("out of randomness, should never happen: %v", err)
	}
	return binary.LittleEndian.Uint64(r[:])
}

// U8 returns a uniformly random uint8.
func U8() uint8 {
	var r [1]uint8
	if _, err := readRandBuf(r[:]); err != nil {
		log.Fatalf("out of randomness, should never happen: %v", err)
	}
	return r[0]
}

// Sign returns +1.0 or -1.0 with equal probabilities.
func Sign() float64 {
	if Boolean() {
		return 1.0
	}
	return -1.0
}

// Boolean returns true or false with equal probability.
func Boolean() bool {
	randBitLock.Lock()
	defer randBitLock.Unlock()
	if randBitPos > 7 { // Out of random bits.
		randBitBuf = U8()
		randBitPos = 0
	}
	res := randBitBuf&(1<<randBitPos) > 0
	randBitPos++
	return res
}

// I63n returns an integer from the set {0,...,n-1} uniformly at random.
// The value of n must be positive.
func I63n(n int64) int64 {
	largestMultipleOfN := (math.MaxInt64 / n) * n
	var positiveRandomInteger int64
	for true {
		// Draw random 64 bit sequence and set sign bit to 0.
		positiveRandomInteger = int64(U64()) & 0x7fffffffffffffff
		if positiveRandomInteger < largestMultipleOfN {
			break
		}
	}
	return positiveRandomInteger % n
}

// Uniform returns a float64 from the interval (0,1] such that each float
// in the interval is returned with positive probability and the resulting
// distribution simulates a continuous uniform distribution on (0, 1].
//
// See http://g/go-nuts/GndbDnHKHuw/VNSrkl9vBQAJ for details.
func Uniform() float64 {
	i := U64() % (1 << 53)
	r := (1 + float64(i)/(1<<53)) / math.Pow(2, Geometric())
	// We want to avoid returning 0, since we're taking the log of the output.
	if r == 0 {
		return 1
	}
	return r
}

// Geometric returns a float64 that counts the number of Bernoulli trials until
// the first success for a success probability of 0.5.
func Geometric() float64 {
	// 1 plus the number of leading zeros from an infinite stream of random bits
	// follows the desired geometric distribution.
	b := 1
	var r uint8
	for r == 0 {
		r = U8()
		b += bits.LeadingZeros8(r)
	}
	return float64(b)
}

// Normal returns a normally distributed float with mean 0 and standard deviation 1.
func Normal() float64 {
	return mathrand.New(&randSource{}).NormFloat64()
}

// randSource implements a cryptographically secure implementation of math.Source.
type randSource struct{}

// Int63 returns a uniformly random int64 in [0, 1<<63).
func (rs randSource) Int63() int64 {
	var r [8]uint8
	if _, err := readRandBuf(r[:]); err != nil {
		log.Fatalf("out of randomness, should never happen: %v", err)
	}
	i := int64(binary.LittleEndian.Uint64(r[:]))
	if i < 0 {
		return -i
	}
	return i
}

// Seed is a no-op.
func (rs randSource) Seed(_ int64) {}
