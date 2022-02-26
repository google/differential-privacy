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

package pbeamtest

import (
	"testing"

	"github.com/google/differential-privacy/privacy-on-beam/pbeam"
	"github.com/google/differential-privacy/privacy-on-beam/pbeam/testutils"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
)

func TestMain(m *testing.M) {
	ptest.Main(m)
}

const (
	// Low ε & δ (i.e. high noise) ensures that we would add noise if test PrivacySpec does not work as intended.
	tinyEpsilon = 1e-10
	tinyDelta   = 1e-200
	// Zero δ is used when public partitions are specified.
	zeroDelta = 0.0
)

// Tests that DistinctPrivacyID bounds per-partition and cross-partition contributions
// correctly, adds no noise and keeps all partitions in test mode.
func TestDistinctPrivacyIDTestMode(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		want                     int64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes to 10 partitions, which implies that count of each
			// partition is 1. With a max contribution of 3, 7 partitions should be dropped. The sum
			// of all counts must then be 3. This also ensures that no partitions (each with a single
			// privacy id) gets thresholded.
			want: 3,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes to 10 partitions, which implies that count of each
			// partition is 1. Contribution bounding is disabled. The sum of all counts must then be 10.
			// This also ensures that no partitions (each with a single privacy id) gets thresholded.
			want: 10,
		},
	} {
		// pairs{privacy_id, partition_key} contains {0,0}, {0,0}, {0,1}, {0,1}, {0,2}, {0,2}, …, {0,9}, {0,9}.
		var pairs []testutils.PairII
		for i := 0; i < 10; i++ {
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...) // Duplicate contributions should be dropped for DistinctPrivacyID.
		}
		wantMetric := []testutils.TestInt64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(pairs, wantMetric)
		col = beam.ParDo(s, testutils.PairToKV, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		got := pbeam.DistinctPrivacyID(s, pcol, pbeam.DistinctPrivacyIDParams{
			MaxPartitionsContributed: tc.maxPartitionsContributed,
			NoiseKind:                pbeam.LaplaceNoise{}})
		counts := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, counts)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("DistinctPrivacyID: %s did not bound contributions correctly: %v", tc.desc, err)
		}
	}
}

// Tests that DistinctPrivacyID with public partitions bounds per-partition and cross-partition
// contributions correctly, adds no noise and respects public partitions (keeps only public partitions)
// in test mode.
func TestDistinctPrivacyIDWithPartitionsTestMode(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		want                     int64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes to 10 partitions, which implies that count of each
			// partition is 1. With a max contribution of 3, 2 out of 5 public partitions should be
			// dropped. The sum of all counts must then be 3.
			want: 3,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes to 10 partitions, which implies that count of each
			// partition is 1. Contribution bounding is disabled and 5 out of 10 partitions are
			// specified as public partitions. The sum of all counts must then be 5.
			want: 5,
		},
	} {
		// pairs{privacy_id, partition_key} contains {0,0}, {0,0}, {0,1}, {0,1}, {0,2}, {0,2}, …, {0,9}, {0,9}.
		var pairs []testutils.PairII
		for i := 0; i < 10; i++ {
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...) // Duplicate contributions should be dropped for DistinctPrivacyID.
		}
		wantMetric := []testutils.TestInt64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(pairs, wantMetric)
		col = beam.ParDo(s, testutils.PairToKV, col)

		partitions := []int{0, 1, 2, 3, 4}
		// Create partition PCollection.
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		got := pbeam.DistinctPrivacyID(s, pcol, pbeam.DistinctPrivacyIDParams{
			MaxPartitionsContributed: tc.maxPartitionsContributed,
			NoiseKind:                pbeam.LaplaceNoise{},
			PublicPartitions:         publicPartitions})
		counts := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, counts)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("DistinctPrivacyID: %s with partitions did not bound contributions correctly: %v", tc.desc, err)
		}
	}
}

// Tests that DistinctPrivacyID with public partitions adds empty partitions not found in the data
// but are in the list of public partitions in test mode.
func TestDistinctPrivacyIDWithPartitionsTestModeAddsEmptyPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc        string
		privacySpec *pbeam.PrivacySpec
	}{
		{
			desc:        "test mode with contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
		},
		{
			desc:        "test mode without contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
		},
	} {
		// pairs{privacy_id, partition_key} contains {0,0}, {0,1}, {0,2}, …, {0,9}.
		var pairs []testutils.PairII
		for i := 0; i < 10; i++ {
			pairs = append(pairs, testutils.PairII{1, i})
		}
		wantMetric := []testutils.TestInt64Metric{
			{9, 1},  // Keep partition 9.
			{10, 0}, // Add partition 10.
		}

		p, s, col, want := ptest.CreateList2(pairs, wantMetric)
		col = beam.ParDo(s, testutils.PairToKV, col)
		partitions := []int{9, 10}

		// Create partition PCollection.
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		got := pbeam.DistinctPrivacyID(s, pcol, pbeam.DistinctPrivacyIDParams{
			MaxPartitionsContributed: 1,
			NoiseKind:                pbeam.LaplaceNoise{},
			PublicPartitions:         publicPartitions})
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("DistinctPrivacyID: %s with partitions did not add empty partitions: %v", tc.desc, err)
		}
	}
}

// Tests that Count bounds per-partition and cross-partition contributions correctly,
// adds no noise and keeps all partitions in test mode.
func TestCountTestMode(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		maxValue                 int64
		want                     int64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			maxValue:                 2,
			// The same privacy ID contributes twice (third contribution is dropped due per-partition
			// contribution bounding) to 10 partitions, which implies that count of each partition is 2.
			// With a max contribution of 3, 7 partitions should be dropped. The sum of all counts must
			// then be 6. This also ensures that no partitions (each with a single privacy id) gets
			// thresholded.
			want: 6,
		},
		{
			desc:                     "test mode without contribution bounding",
			maxPartitionsContributed: 3,
			maxValue:                 2,
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			// The same privacy ID contributes thrice to 10 partitions, which implies that count of each
			// partition is 3. Contribution bounding is disabled. The sum of all counts must then be 30.
			// This also ensures that no partitions (each with a single privacy id) gets thresholded.
			want: 30,
		},
	} {
		// pairs{privacy_id, partition_key} contains {0,0}, {0,0}, {0,0}, {0,1}, {0,1}, {0,1}, {0,2}, {0,2}, {0,2}, …, {0,9}, {0,9}, {0,9}.
		var pairs []testutils.PairII
		for i := 0; i < 10; i++ {
			// When contribution bounding is enabled, one of the three contributions should be dropped since MaxValue is 2.
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
		}
		wantMetric := []testutils.TestInt64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(pairs, wantMetric)
		col = beam.ParDo(s, testutils.PairToKV, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		got := pbeam.Count(s, pcol, pbeam.CountParams{
			MaxValue:                 tc.maxValue,
			MaxPartitionsContributed: tc.maxPartitionsContributed,
			NoiseKind:                pbeam.LaplaceNoise{}})
		counts := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, counts)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("Count: %s did not bound contributions correctly: %v", tc.desc, err)
		}
	}
}

// Tests that Count with public partitions bounds per-partition and cross-partition contributions
// correctly, adds no noise and respects public partitions (keeps only public partitions) in test
// mode.
func TestCountWithPartitionsTestMode(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		maxValue                 int64
		want                     int64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			maxValue:                 2,
			// The same privacy ID contributes twice (third contribution is dropped due per-partition
			// contribution bounding) to 10 partitions, which implies that count of each partition is 2.
			// With a max contribution of 3, 2 out of 5 public partitions should be dropped. The sum of
			// all counts must then be 6.
			want: 6,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			maxValue:                 2,
			// The same privacy ID contributes thrice to 10 partitions, which implies that count of each
			// partition is 3. Contribution bounding is disabled and 5 out of 10 partitions are specified
			// as public partitions. The sum of all counts must then be 15.
			want: 15,
		},
	} {
		// pairs{privacy_id, partition_key} contains {0,0}, {0,0}, {0,0}, {0,1}, {0,1}, {0,1}, {0,2}, {0,2}, {0,2}, …, {0,9}, {0,9}, {0,9}.
		var pairs []testutils.PairII
		for i := 0; i < 10; i++ {
			// When contribution bounding is enabled, one of the three contributions should be dropped since MaxValue is 2.
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
		}
		wantMetric := []testutils.TestInt64Metric{
			{0, tc.want},
		}

		p, s, col, want := ptest.CreateList2(pairs, wantMetric)
		col = beam.ParDo(s, testutils.PairToKV, col)
		partitions := []int{0, 1, 2, 3, 4}
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		got := pbeam.Count(s, pcol, pbeam.CountParams{
			MaxValue:                 tc.maxValue,
			MaxPartitionsContributed: tc.maxPartitionsContributed,
			NoiseKind:                pbeam.LaplaceNoise{},
			PublicPartitions:         publicPartitions})
		counts := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, counts)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("Count: %s with partitions did not bound contributions correctly: %v", tc.desc, err)
		}
	}
}

// Tests that Count with public partitions adds empty partitions not found in the data
// but are in the list of public partitions in test mode.
func TestCountWithPartitionsTestModeAddsEmptyPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc        string
		privacySpec *pbeam.PrivacySpec
	}{
		{
			desc:        "test mode with contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
		},
		{
			desc:        "test mode without contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
		},
	} {
		var pairs []testutils.PairII
		for i := 0; i < 10; i++ {
			pairs = append(pairs, testutils.PairII{1, i})
		}
		wantMetric := []testutils.TestInt64Metric{
			{9, 1},  // Keep partition 9.
			{10, 0}, // Add partition 10.
		}

		p, s, col, want := ptest.CreateList2(pairs, wantMetric)
		col = beam.ParDo(s, testutils.PairToKV, col)
		partitions := []int{9, 10}
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		got := pbeam.Count(s, pcol, pbeam.CountParams{
			MaxValue:                 2,
			MaxPartitionsContributed: 1,
			NoiseKind:                pbeam.LaplaceNoise{},
			PublicPartitions:         publicPartitions})
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("Count: %s with partitions did not add empty partitions: %v", tc.desc, err)
		}
	}
}

// Tests that SumPerKey bounds per-partition and cross-partition contributions correctly,
// adds no noise and keeps all partitions in test mode with ints.
func TestSumPerKeyTestModeInt(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		minValue                 float64
		maxValue                 float64
		want                     int64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			minValue:                 0.0,
			maxValue:                 1.0,
			// The same privacy ID contributes "1" ("2" is clamped to "1" since MaxValue is 1) to 10
			// partitions, which implies that sum of each partition is 1. With a max contribution of 3,
			// 7 partitions should be dropped. The sum of all sum must then be 3. This also ensures that
			// no partitions (each with a single privacy id) gets thresholded.
			want: 3,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			minValue:                 0.0,
			maxValue:                 1.0,
			// The same privacy ID contributes "2" to 10 partitions, which implies that sum of each
			// partition is 2. Contribution bounding is disabled. The sum of all sums must then be 20.
			// This also ensures that no partitions (each with a single privacy id) gets thresholded.
			want: 20,
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,1}, {0,0,1}, {0,1,1}, {0,0,1}, {0,2,1}, {0,2,1}, …, {0,9,1}, {0,9,1}.
		var triples []testutils.TripleWithIntValue
		for i := 0; i < 10; i++ {
			// The sum total contributions per user per partition is 2. When contribution bounding is
			// enabled, this will be clamped to 1 since MaxValue is 1.
			triples = append(triples, testutils.MakeSampleTripleWithIntValue(1, i)...)
			triples = append(triples, testutils.MakeSampleTripleWithIntValue(1, i)...)
		}
		wantMetric := []testutils.TestInt64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		got := pbeam.SumPerKey(s, pcol, pbeam.SumParams{
			MaxPartitionsContributed: tc.maxPartitionsContributed,
			MinValue:                 tc.minValue,
			MaxValue:                 tc.maxValue,
			NoiseKind:                pbeam.LaplaceNoise{}})
		sums := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, sums)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey: %s did not bound contributions correctly for ints: %v", tc.desc, err)
		}
	}
}

// Tests that SumPerKey with public partitions bounds per-partition and cross-partition contributions
// correctly, adds no noise and respects public partitions (keeps only public partitions) in test
// mode with ints.
func TestSumPerKeyWithPartitionsTestModeInt(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		minValue                 float64
		maxValue                 float64
		want                     int64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			minValue:                 0.0,
			maxValue:                 1.0,
			// The same privacy ID contributes "1" ("2" is clamped to "1" since MaxValue is 1) to 10
			// partitions, which implies that sum of each partition is 1. With a max contribution of 3,
			// 2 out of 5 public partitions should be dropped. The sum of all sums must then be 3.
			want: 3,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			minValue:                 0.0,
			maxValue:                 1.0,
			// The same privacy ID contributes "2" to 10 partitions, which implies that sum of each
			// partition is 2. Contribution bounding is disabled and 5 out of 10 partitions are specified
			// as public partitions. The sum of all sums must then be 10.
			want: 10,
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,1}, {0,0,1}, {0,1,1}, {0,0,1}, {0,2,1}, {0,2,1}, …, {0,9,1}, {0,9,1}.
		var triples []testutils.TripleWithIntValue
		for i := 0; i < 10; i++ {
			// The sum total contributions per user per partition is 2. When contribution bounding is
			// enabled, this will be clamped to 1 since MaxValue is 1.
			triples = append(triples, testutils.MakeSampleTripleWithIntValue(1, i)...)
			triples = append(triples, testutils.MakeSampleTripleWithIntValue(1, i)...)
		}
		wantMetric := []testutils.TestInt64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)
		partitions := []int{0, 1, 2, 3, 4}
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		got := pbeam.SumPerKey(s, pcol, pbeam.SumParams{
			MaxPartitionsContributed: tc.maxPartitionsContributed,
			MinValue:                 tc.minValue,
			MaxValue:                 tc.maxValue,
			NoiseKind:                pbeam.LaplaceNoise{},
			PublicPartitions:         publicPartitions})
		sums := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, sums)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey: %s with partitions did not bound contributions correctly for ints: %v", tc.desc, err)
		}
	}
}

// Tests that SumPerKey with public partitions adds empty partitions not found in the
// data but are in the list of public partitions in test mode with ints.
func TestSumPerKeyWithPartitionsTestModeAddsEmptyPartitionsInt(t *testing.T) {
	for _, tc := range []struct {
		desc        string
		privacySpec *pbeam.PrivacySpec
	}{
		{
			desc:        "test mode with contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
		},
		{
			desc:        "test mode without contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,1}, {0,1,1}, {0,2,1}, …, {0,9,1}.
		var triples []testutils.TripleWithIntValue
		for i := 0; i < 10; i++ {
			triples = append(triples, testutils.MakeSampleTripleWithIntValue(1, i)...)
		}
		wantMetric := []testutils.TestInt64Metric{
			{9, 1},  // Keep partition 9.
			{10, 0}, // Add partition 10.
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)
		partitions := []int{9, 10}
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		got := pbeam.SumPerKey(s, pcol, pbeam.SumParams{
			MaxPartitionsContributed: 1,
			MinValue:                 0,
			MaxValue:                 1,
			NoiseKind:                pbeam.LaplaceNoise{},
			PublicPartitions:         publicPartitions})
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey: %s with partitions did not add empty partitions for ints: %v", tc.desc, err)
		}
	}
}

// Tests that SumPerKey bounds per-partition and cross-partition contributions correctly,
// adds no noise and keeps all partitions in test mode with floats.
func TestSumPerKeyTestModeFloat(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		minValue                 float64
		maxValue                 float64
		want                     float64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			minValue:                 0.0,
			maxValue:                 1.0,
			// The same privacy ID contributes "1.0" ("2.0" is clamped to "1.0" since MaxValue is 1) to 10
			// partitions, which implies that sum of each partition is 1.0. With a max contribution of 3,
			// 7 partitions should be dropped. The sum of all sum must then be 3.0. This also ensures that
			// no partitions (each with a single privacy id) gets thresholded.
			want: 3.0,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			minValue:                 0.0,
			maxValue:                 1.0,
			// The same privacy ID contributes "2.0" to 10 partitions, which implies that sum of each
			// partition is 2.0. Contribution bounding is disabled. The sum of all sums must then be 20.0.
			// This also ensures that no partitions (each with a single privacy id) gets thresholded.
			want: 20.0,
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,1}, {0,0,1}, {0,1,1}, {0,0,1}, {0,2,1}, {0,2,1}, …, {0,9,1}, {0,9,1}.
		var triples []testutils.TripleWithFloatValue
		for i := 0; i < 10; i++ {
			// The sum total contributions per user per partition is 2.0. When contribution bounding is
			// enabled, this will be clamped to 1.0 since MaxValue is 1.
			triples = append(triples, testutils.MakeSampleTripleWithFloatValue(1, i)...)
			triples = append(triples, testutils.MakeSampleTripleWithFloatValue(1, i)...)
		}
		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.SumPerKey(s, pcol, pbeam.SumParams{
			MaxPartitionsContributed: tc.maxPartitionsContributed,
			MinValue:                 tc.minValue,
			MaxValue:                 tc.maxValue,
			NoiseKind:                pbeam.LaplaceNoise{}})
		sums := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, sums)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		if err := testutils.EqualsKVFloat64(s, got, want); err != nil {
			t.Fatalf("EqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey: %s did not bound contributions correctly for floats: %v", tc.desc, err)
		}
	}
}

// Tests that SumPerKey with public partitions bounds per-partition and cross-partition contributions
// correctly, adds no noise and respects public partitions (keeps only public partitions) in test
// mode with floats.
func TestSumPerKeyWithPartitionsTestModeFloat(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		minValue                 float64
		maxValue                 float64
		want                     float64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			minValue:                 0.0,
			maxValue:                 1.0,
			// The same privacy ID contributes "1.0" ("2.0" is clamped to "1.0" since MaxValue is 1) to 10
			// partitions, which implies that sum of each partition is 1.0. With a max contribution of 3,
			// 2 out of 5 public partitions should be dropped. The sum of all sums must then be 3.0.
			want: 3.0,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			minValue:                 0.0,
			maxValue:                 1.0,
			// The same privacy ID contributes "2.0" to 10 partitions, which implies that sum of each
			// partition is 2. Contribution bounding is disabled and 5 out of 10 partitions are specified
			// as public partitions. The sum of all sums must then be 10.0.
			want: 10.0,
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,1}, {0,0,1}, {0,1,1}, {0,0,1}, {0,2,1}, {0,2,1}, …, {0,9,1}, {0,9,1}.
		var triples []testutils.TripleWithFloatValue
		for i := 0; i < 10; i++ {
			// The sum total contributions per user per partition is 2.0. When contribution bounding is
			// enabled, this will be clamped to 1.0 since MaxValue is 1.
			triples = append(triples, testutils.MakeSampleTripleWithFloatValue(1, i)...)
			triples = append(triples, testutils.MakeSampleTripleWithFloatValue(1, i)...)
		}
		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
		partitions := []int{0, 1, 2, 3, 4}
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.SumPerKey(s, pcol, pbeam.SumParams{
			MaxPartitionsContributed: tc.maxPartitionsContributed,
			MinValue:                 tc.minValue,
			MaxValue:                 tc.maxValue,
			NoiseKind:                pbeam.LaplaceNoise{},
			PublicPartitions:         publicPartitions})
		sums := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, sums)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		if err := testutils.EqualsKVFloat64(s, got, want); err != nil {
			t.Fatalf("EqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey: %s with partitions did not bound contributions correctly for floats: %v", tc.desc, err)
		}
	}
}

// Tests that SumPerKey with public partitions adds empty partitions not found in the
// data but are in the list of public partitions in test mode with floats.
func TestSumPerKeyWithPartitionsTestModeAddsEmptyPartitionsFloat(t *testing.T) {
	for _, tc := range []struct {
		desc        string
		privacySpec *pbeam.PrivacySpec
	}{
		{
			desc:        "test mode with contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
		},
		{
			desc:        "test mode without contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,1}, {0,1,1}, {0,2,1}, …, {0,9,1}.
		var triples []testutils.TripleWithFloatValue
		for i := 0; i < 10; i++ {
			triples = append(triples, testutils.MakeSampleTripleWithFloatValue(1, i)...)
		}
		wantMetric := []testutils.TestFloat64Metric{
			{9, 1.0},  // Keep partition 9.
			{10, 0.0}, // Add partition 10.
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
		partitions := []int{9, 10}
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.SumPerKey(s, pcol, pbeam.SumParams{MaxPartitionsContributed: 1,
			MinValue:         0,
			MaxValue:         1,
			NoiseKind:        pbeam.LaplaceNoise{},
			PublicPartitions: publicPartitions})
		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		if err := testutils.EqualsKVFloat64(s, got, want); err != nil {
			t.Fatalf("EqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("SumPerKey: %s with partitions did not add empty partitions for floats: %v", tc.desc, err)
		}
	}
}

// Tests that MeanPerKey bounds cross-partition contributions correctly, adds no noise
// and keeps all partitions in test mode.
func TestMeanPerKeyTestModeCrossPartitionContributionBounding(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		want                     float64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes "1.0" to 10 partitions, which implies that mean of each
			// partition is 1.0. With a max contribution of 3, 7 partitions should be dropped. The sum
			// of all means must then be 3.0. This also ensures that no partitions (each with a single
			// privacy id) gets thresholded.
			want: 3.0,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes "1.0" to 10 partitions, which implies that mean of each
			// partition is 1.0. Cross-partition contribution bounding is disabled. The sum of all means
			// must then be 10.0. This also ensures that no partitions (each with a single privacy id)
			// gets thresholded.
			want: 10.0,
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,1}, {0,1,1}, {0,2,1}, …, {0,9,1}.
		var triples []testutils.TripleWithFloatValue
		for i := 0; i < 10; i++ {
			triples = append(triples, testutils.MakeSampleTripleWithFloatValue(1, i)...)
		}
		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.MeanPerKey(s, pcol, pbeam.MeanParams{
			MaxPartitionsContributed:     tc.maxPartitionsContributed,
			MaxContributionsPerPartition: 1,
			MinValue:                     0,
			MaxValue:                     1,
			NoiseKind:                    pbeam.LaplaceNoise{}})
		means := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, means)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		if err := testutils.EqualsKVFloat64(s, got, want); err != nil {
			t.Fatalf("EqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("MeanPerKey: %s did not do cross-partition contribution bounding correctly: %v", tc.desc, err)
		}
	}
}

// Tests that MeanPerKey bounds per-partition contributions correctly, adds no noise
// and keeps all partitions in test mode.
func TestMeanPerKeyTestModePerPartitionContributionBounding(t *testing.T) {
	for _, tc := range []struct {
		desc                        string
		privacySpec                 *pbeam.PrivacySpec
		maxContributionPerPartition int64
		minValue                    float64
		maxValue                    float64
		want                        float64
	}{
		{
			desc:                        "test mode with contribution bounding",
			privacySpec:                 NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxContributionPerPartition: 1,
			minValue:                    0.0,
			maxValue:                    50.0,
			// MaxContributionsPerPartition = 1, but id = 0 contributes 3 times to partition 0.
			// There will be a per-partition contribution bounding stage.
			// In this stage the algorithm will arbitrarily chose one of these 3 contributions.
			// The mean should be equal to 50/50 = 1.0 (not 150/52 ≈ 2.88, if no per-partition contribution bounding is done).
			want: 1.0,
		},
		{
			desc:                        "test mode without contribution bounding",
			privacySpec:                 NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxContributionPerPartition: 1,
			minValue:                    0.0,
			maxValue:                    50.0,
			// There will not be a per-partition contribution bounding stage.
			// The mean should be equal to 150/52 = 2.88461538462.
			want: 2.88461538462,
		},
	} {
		var triples []testutils.TripleWithFloatValue
		// triples{privacy_id, partition_key, value} contains {0,0,50}, {0,0,50}, {0,0,50}, {1,0,0}, {2,0,0},…, {49,0,0}.
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
		triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(1, 49, 0, 0)...)
		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.MeanPerKey(s, pcol, pbeam.MeanParams{
			MaxPartitionsContributed:     3,
			MaxContributionsPerPartition: tc.maxContributionPerPartition,
			MinValue:                     tc.minValue,
			MaxValue:                     tc.maxValue,
			NoiseKind:                    pbeam.LaplaceNoise{}})
		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		tolerance := 1e-10 // Using a small tolerance to make up for the rounding errors due to summation & division.
		if err := testutils.ApproxEqualsKVFloat64(s, got, want, tolerance); err != nil {
			t.Fatalf("ApproxEqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("MeanPerKey: %s did not do per-partition contribution bounding correctly: %v", tc.desc, err)
		}
	}
}

// Tests that MeanPerKey with public partitions bounds cross-partition contributions correctly,
// adds no noise and respects public partitions (keeps only public partitions) in test mode.
func TestMeanPerKeyWithPartitionsTestModeCrossPartitionContributionBounding(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		want                     float64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes "1.0" to 10 partitions, which implies that mean of each
			// partition is 1.0. With a max contribution of 3, 2 out of 5 public partitions should be
			// dropped. The sum of all means must then be 3.0.
			want: 3.0,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes "1.0" to 10 partitions, which implies that mean of each
			// partition is 1.0. Cross-partition contribution bounding is disabled and 5 out of 10 partitions
			// are specified as public partitions. The sum of all means must then be 5.0.
			want: 5.0,
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,1}, {0,1,1}, {0,2,1},…, {0,9,1}.
		var triples []testutils.TripleWithFloatValue
		for i := 0; i < 10; i++ {
			triples = append(triples, testutils.MakeSampleTripleWithFloatValue(1, i)...)
		}
		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
		partitions := []int{0, 1, 2, 3, 4}
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		// Setting MinValue to -1 and MaxValue to 1 so that empty partitions have a mean of 0.
		got := pbeam.MeanPerKey(s, pcol, pbeam.MeanParams{MaxPartitionsContributed: tc.maxPartitionsContributed,
			MaxContributionsPerPartition: 1,
			MinValue:                     -1,
			MaxValue:                     1,
			NoiseKind:                    pbeam.LaplaceNoise{},
			PublicPartitions:             publicPartitions})
		means := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, means)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		if err := testutils.EqualsKVFloat64(s, got, want); err != nil {
			t.Fatalf("EqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("MeanPerKey: %s with partitions did not do cross-partition contribution bounding correctly: %v", tc.desc, err)
		}
	}
}

// Tests that MeanPerKey with public partitions bounds per-partition contributions correctly,
// adds no noise and respects public partitions (keeps public partitions and adds empty
// partitions) in test mode.
func TestMeanPerKeyWithPartitionsTestModePerPartitionContributionBoundingAddsEmptyPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc                        string
		privacySpec                 *pbeam.PrivacySpec
		maxContributionPerPartition int64
		minValue                    float64
		maxValue                    float64
		want                        float64
	}{
		{
			desc:                        "test mode with contribution bounding",
			privacySpec:                 NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
			maxContributionPerPartition: 1,
			minValue:                    0.0,
			maxValue:                    50.0,
			// MaxContributionsPerPartition = 1, but id = 0 contributes 3 times to partition 0.
			// There will be a per-partition contribution bounding stage.
			// In this stage the algorithm will arbitrarily chose one of these 3 contributions.
			// The mean should be equal to 50/50 = 1.0 (not 150/52 ≈ 2.88, if no per-partition contribution bounding is done).
			want: 1.0,
		},
		{
			desc:                        "test mode without contribution bounding",
			privacySpec:                 NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
			maxContributionPerPartition: 1,
			minValue:                    0.0,
			maxValue:                    50.0,
			// There will not be a per-partition contribution bounding stage.
			// The mean should be equal to 150/52 = 2.88461538462.
			want: 2.88461538462,
		},
	} {
		var triples []testutils.TripleWithFloatValue
		// triples{privacy_id, partition_key, value} contains {0,0,50}, {0,0,50}, {0,0,50}, {1,0,0}, {2,0,0},…, {49,0,0}.
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
		triples = append(triples, testutils.MakeTripleWithFloatValue(1, 0, 50)...)
		triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(1, 49, 0, 0)...)

		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
			{1, 25.0}, // Empty partition (output is midpoint of MinValue and MaxValue).
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
		partitions := []int{0, 1}
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.MeanPerKey(s, pcol, pbeam.MeanParams{
			MaxPartitionsContributed:     1,
			MaxContributionsPerPartition: tc.maxContributionPerPartition,
			MinValue:                     tc.minValue,
			MaxValue:                     tc.maxValue,
			NoiseKind:                    pbeam.LaplaceNoise{},
			PublicPartitions:             publicPartitions})
		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		tolerance := 1e-10 // Using a small tolerance to make up for the rounding errors due to summation & division.
		if err := testutils.ApproxEqualsKVFloat64(s, got, want, tolerance); err != nil {
			t.Fatalf("EqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("MeanPerKey: %s with partitions did not do per-partition contribution bounding correctly or added empty partitions: %v", tc.desc, err)
		}
	}
}

// Tests that QuantilesPerKey bounds cross-partitions contributions correctly, adds no
// noise and keeps all partitions in test mode.
func TestQuantilesPerKeyTestModeCrossPartitionContributionBounding(t *testing.T) {
	for _, tc := range []struct {
		desc                         string
		privacySpec                  *pbeam.PrivacySpec
		maxContributionsPerPartition int64
		maxPartitionsContributed     int64
		minValue                     float64
		maxValue                     float64
		want                         float64
	}{
		{
			desc:                         "test mode with contribution bounding",
			privacySpec:                  NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxContributionsPerPartition: 20,
			maxPartitionsContributed:     1,
			minValue:                     0.0,
			maxValue:                     1.0,
			// 10 distinct privacy IDs contribute 0.0 to partition 0 and another 10 distinct
			// privacy IDs contribute 0.0 to partition 1. A single privacy ID (different from
			// these 20 privacy IDs) then contributes 20 "1.0"s to both partition 0 and partition 1.
			// MaxPartitionsContributed is 1, so contributions to only one of these partitions will
			// be kept. The median (rank=0.50) of one of these partitions must then be 0.0 and the other
			// 1.0. The sum of these medians must then equal 1.0 (as opposed to 2.0 if no contribution
			// bounding takes place). This also ensures that no partitions get thresholded.
			want: 1.0,
		},
		{
			desc:                         "test mode without contribution bounding",
			privacySpec:                  NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxContributionsPerPartition: 20,
			maxPartitionsContributed:     1,
			minValue:                     0.0,
			maxValue:                     1.0,
			// 10 distinct privacy IDs contribute 0.0 to partition 0 and another 10 distinct
			// privacy IDs contribute 0.0 to partition 1. A single privacy ID (different from
			// these 20 privacy IDs) then contributes 20 "1.0"s to both partition 0 and partition 1.
			// Cross-partition contribution bounding is disabled, so the median (rank=0.50) of both of
			// these partitions must then be 1.0. The sum of these medians must then equal 2.0.
			want: 2.0,
		},
	} {
		triples := testutils.ConcatenateTriplesWithFloatValue(
			testutils.MakeTripleWithFloatValue(10, 0, 0.0),
			testutils.MakeTripleWithFloatValueStartingFromKey(10, 10, 1, 0.0))
		for i := 0; i < 20; i++ {
			triples = append(triples, testutils.TripleWithFloatValue{ID: 20, Partition: 0, Value: 1.0})
			triples = append(triples, testutils.TripleWithFloatValue{ID: 20, Partition: 1, Value: 1.0})
		}

		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
		}

		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.QuantilesPerKey(s, pcol, pbeam.QuantilesParams{
			MaxContributionsPerPartition: tc.maxContributionsPerPartition,
			MaxPartitionsContributed:     tc.maxPartitionsContributed,
			MinValue:                     tc.minValue,
			MaxValue:                     tc.maxValue,
			Ranks:                        []float64{0.50},
			NoiseKind:                    pbeam.LaplaceNoise{}})
		got = beam.ParDo(s, testutils.DereferenceFloat64Slice, got)
		medians := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, medians)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		// Tolerance is multiplied by 2 because we sum over 2 partitions.
		tolerance := QuantilesTolerance(tc.minValue, tc.maxValue) * 2
		if err := testutils.ApproxEqualsKVFloat64(s, got, want, tolerance); err != nil {
			t.Fatalf("ApproxEqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("QuantilesPerKey: %s did not bound cross-partition contributions correctly: %v", tc.desc, err)
		}
	}
}

// Tests that QuantilesPerKey with partitions bounds cross-partitions contributions correctly
// and adds no noise in test mode.
func TestQuantilesPerKeyWithPartitionsTestModeCrossPartitionContributionBounding(t *testing.T) {
	for _, tc := range []struct {
		desc                         string
		privacySpec                  *pbeam.PrivacySpec
		maxContributionsPerPartition int64
		maxPartitionsContributed     int64
		minValue                     float64
		maxValue                     float64
		want                         float64
	}{
		{
			desc:                         "test mode with contribution bounding",
			privacySpec:                  NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
			maxContributionsPerPartition: 20,
			maxPartitionsContributed:     1,
			minValue:                     0.0,
			maxValue:                     1.0,
			// 10 distinct privacy IDs contribute 0.0 to partition 0 and another 10 distinct
			// privacy IDs contribute 0.0 to partition 1. A single privacy ID (different from
			// these 20 privacy IDs) then contributes 20 "1.0"s to both partition 0 and partition 1.
			// MaxPartitionsContributed is 1, so contributions to only one of these partitions will
			// be kept. The median (rank=0.50) of one of these partitions must then be 0.0 and the other
			// 1.0. The sum of these medians must then equal 1.0 (as opposed to 2.0 if no contribution
			// bounding takes place).
			want: 1.0,
		},
		{
			desc:                         "test mode without contribution bounding",
			privacySpec:                  NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
			maxContributionsPerPartition: 20,
			maxPartitionsContributed:     1,
			minValue:                     0.0,
			maxValue:                     1.0,
			// 10 distinct privacy IDs contribute 0.0 to partition 0 and another 10 distinct
			// privacy IDs contribute 0.0 to partition 1. A single privacy ID (different from
			// these 20 privacy IDs) then contributes 20 "1.0"s to both partition 0 and partition 1.
			// Cross-partition contribution bounding is disabled, so the median (rank=0.50) of both of
			// these partitions must then be 1.0. The sum of these medians must then equal 2.0.
			want: 2.0,
		},
	} {
		triples := testutils.ConcatenateTriplesWithFloatValue(
			testutils.MakeTripleWithFloatValue(10, 0, 0.0),
			testutils.MakeTripleWithFloatValueStartingFromKey(10, 10, 1, 0.0))
		for i := 0; i < 20; i++ {
			triples = append(triples, testutils.TripleWithFloatValue{ID: 200, Partition: 0, Value: 1.0})
			triples = append(triples, testutils.TripleWithFloatValue{ID: 200, Partition: 1, Value: 1.0})
		}

		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
		partitions := []int{0, 1}
		publicPartitions := beam.CreateList(s, partitions)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.QuantilesPerKey(s, pcol, pbeam.QuantilesParams{
			MaxContributionsPerPartition: tc.maxContributionsPerPartition,
			MaxPartitionsContributed:     tc.maxPartitionsContributed,
			MinValue:                     tc.minValue,
			MaxValue:                     tc.maxValue,
			Ranks:                        []float64{0.50},
			PublicPartitions:             publicPartitions,
			NoiseKind:                    pbeam.LaplaceNoise{}})
		got = beam.ParDo(s, testutils.DereferenceFloat64Slice, got)
		medians := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, medians)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		// Tolerance is multiplied by 2 because we sum over 2 partitions.
		tolerance := QuantilesTolerance(tc.minValue, tc.maxValue) * 2
		if err := testutils.ApproxEqualsKVFloat64(s, got, want, tolerance); err != nil {
			t.Fatalf("ApproxEqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("QuantilesPerKey: %s with partitions did not bound cross-partition contributions correctly: %v", tc.desc, err)
		}
	}
}

// Tests that QuantilesPerKey bounds per-partition contributions correctly, adds no noise
// and keeps all partitions in test mode.
func TestQuantilesPerKeyTestModePerPartitionContributionBounding(t *testing.T) {
	for _, tc := range []struct {
		desc                        string
		privacySpec                 *pbeam.PrivacySpec
		maxContributionPerPartition int64
		minValue                    float64
		maxValue                    float64
		want                        float64
	}{
		{
			desc:                        "test mode with contribution bounding",
			privacySpec:                 NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxContributionPerPartition: 1,
			minValue:                    0.0,
			maxValue:                    1.0,
			// First 50 privacy IDs contribute "0.0" 3 times to partition 0 and the next 50 privacy IDs
			// contribute "1.0" to the same partition.
			// There will be a per-partition contribution bounding stage. MaxContributionsPerPartition=1, so
			// the algorithm will arbitrarily keep one of these 3 contributions for the first 50 privacy IDs.
			// There will be equal number of "0.0"s and "1.0", so rank=0.6 should be equal to 1.0 (not 0.0,
			// if no per-partition contribution bounding is done)
			want: 1.0,
		},
		{
			desc:                        "test mode without contribution bounding",
			privacySpec:                 NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxContributionPerPartition: 1,
			minValue:                    0.0,
			maxValue:                    1.0,
			// First 50 privacy IDs contribute "0.0" 3 times to partition 0 and the next 50 privacy IDs
			// contribute "1.0" to the same partition.
			// There will not be a per-partition contribution bounding stage, meaning that there will be 150
			// "0.0"s and 50 "1.0"s. rank=0.6 should be equal to 0.0.
			want: 0.0,
		},
	} {
		var triples []testutils.TripleWithFloatValue
		// triples{privacy_id, partition_key, value} contains {0,0,0}, {0,0,0}, {0,0,0}, …, {49,0,0}, {49,0,0}, {49,0,0}, {50,0,1}, {51,0,1}, …, {99, 0, 1}.
		triples = append(triples, testutils.MakeTripleWithFloatValue(50, 0, 0.0)...)
		triples = append(triples, testutils.MakeTripleWithFloatValue(50, 0, 0.0)...)
		triples = append(triples, testutils.MakeTripleWithFloatValue(50, 0, 0.0)...)
		triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(50, 50, 0, 1.0)...)
		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.QuantilesPerKey(s, pcol, pbeam.QuantilesParams{
			MaxContributionsPerPartition: tc.maxContributionPerPartition,
			MaxPartitionsContributed:     1,
			MinValue:                     tc.minValue,
			MaxValue:                     tc.maxValue,
			Ranks:                        []float64{0.6},
			NoiseKind:                    pbeam.LaplaceNoise{}})
		got = beam.ParDo(s, testutils.DereferenceFloat64Slice, got)

		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		if err := testutils.ApproxEqualsKVFloat64(s, got, want, QuantilesTolerance(tc.minValue, tc.maxValue)); err != nil {
			t.Fatalf("ApproxEqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("QuantilesPerKey: %s did not do per-partition contribution bounding correctly: %v", tc.desc, err)
		}
	}
}

// Tests that QuantilesPerKey with partition bounds per-partition contributions correctly,
// adds no noise and keeps all partitions in test mode.
func TestQuantilesPerKeyWithPartitionsTestModePerPartitionContributionBounding(t *testing.T) {
	for _, tc := range []struct {
		desc                        string
		privacySpec                 *pbeam.PrivacySpec
		maxContributionPerPartition int64
		minValue                    float64
		maxValue                    float64
		want                        float64
	}{
		{
			desc:                        "test mode with contribution bounding",
			privacySpec:                 NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
			maxContributionPerPartition: 1,
			minValue:                    0.0,
			maxValue:                    1.0,
			// First 50 privacy IDs contribute "0.0" 3 times to partition 0 and the next 50 privacy IDs
			// contribute "1.0" to the same partition.
			// There will be a per-partition contribution bounding stage. MaxContributionsPerPartition=1, so
			// the algorithm will arbitrarily keep one of these 3 contributions for the first 50 privacy IDs.
			// There will be equal number of "0.0"s and "1.0", so rank=0.6 should be equal to 1.0 (not 0.0,
			// if no per-partition contribution bounding is done)
			want: 1.0,
		},
		{
			desc:                        "test mode without contribution bounding",
			privacySpec:                 NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
			maxContributionPerPartition: 1,
			minValue:                    0.0,
			maxValue:                    1.0,
			// First 50 privacy IDs contribute "0.0" 3 times to partition 0 and the next 50 privacy IDs
			// contribute "1.0" to the same partition.
			// There will not be a per-partition contribution bounding stage, meaning that there will be 150
			// "0.0"s and 50 "1.0"s. rank=0.6 should be equal to 0.0.
			want: 0.0,
		},
	} {
		var triples []testutils.TripleWithFloatValue
		// triples{privacy_id, partition_key, value} contains {0,0,0}, {0,0,0}, {0,0,0}, …, {49,0,0}, {49,0,0}, {49,0,0}, {50,0,1}, {51,0,1}, …, {99, 0, 1}.
		triples = append(triples, testutils.MakeTripleWithFloatValue(50, 0, 0.0)...)
		triples = append(triples, testutils.MakeTripleWithFloatValue(50, 0, 0.0)...)
		triples = append(triples, testutils.MakeTripleWithFloatValue(50, 0, 0.0)...)
		triples = append(triples, testutils.MakeTripleWithFloatValueStartingFromKey(50, 50, 0, 1.0)...)
		wantMetric := []testutils.TestFloat64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)
		publicPartitions := beam.CreateList(s, []int{0})

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.QuantilesPerKey(s, pcol, pbeam.QuantilesParams{
			MaxContributionsPerPartition: tc.maxContributionPerPartition,
			MaxPartitionsContributed:     1,
			MinValue:                     tc.minValue,
			MaxValue:                     tc.maxValue,
			Ranks:                        []float64{0.6},
			PublicPartitions:             publicPartitions,
			NoiseKind:                    pbeam.LaplaceNoise{}})
		got = beam.ParDo(s, testutils.DereferenceFloat64Slice, got)

		want = beam.ParDo(s, testutils.Float64MetricToKV, want)
		if err := testutils.ApproxEqualsKVFloat64(s, got, want, QuantilesTolerance(tc.minValue, tc.maxValue)); err != nil {
			t.Fatalf("ApproxEqualsKVFloat64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("QuantilesPerKey: %s with partitions did not do per-partition contribution bounding correctly: %v", tc.desc, err)
		}
	}
}

// Checks that QuantilesPerKey with partitions applies public partitions correctly in test mode.
func TestQuantilesPerKeyWithPartitionsAppliesPublicPartitions(t *testing.T) {
	for _, tc := range []struct {
		desc        string
		privacySpec *pbeam.PrivacySpec
	}{
		{
			desc:        "test mode with contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, zeroDelta),
		},
		{
			desc:        "test mode without contribution bounding",
			privacySpec: NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, zeroDelta),
		},
	} {
		triples := testutils.ConcatenateTriplesWithFloatValue(
			testutils.MakeTripleWithFloatValue(100, 0, 1.0),
			testutils.MakeTripleWithFloatValue(100, 0, 4.0),
			testutils.MakeTripleWithFloatValueStartingFromKey(100, 100, 1, 1.0))

		wantMetric := []testutils.TestFloat64SliceMetric{
			{0, []float64{1.0, 1.0, 4.0, 4.0}},
			// Partition 1 is not in the list of public partitions, so it will be dropped.
			{2, []float64{0.5, 1.25, 3.75, 4.5}}, // Empty partition is linearly interpolated.
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithFloatValue, col)

		lower := 0.0
		upper := 5.0
		ranks := []float64{0.10, 0.25, 0.75, 0.90}
		publicPartitions := beam.CreateList(s, []int{0, 2})

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithFloatValueToKV, pcol)
		got := pbeam.QuantilesPerKey(s, pcol, pbeam.QuantilesParams{
			MaxPartitionsContributed:     1,
			MaxContributionsPerPartition: 2,
			MinValue:                     lower,
			MaxValue:                     upper,
			Ranks:                        ranks,
			PublicPartitions:             publicPartitions,
		})

		want = beam.ParDo(s, testutils.Float64SliceMetricToKV, want)
		if err := testutils.ApproxEqualsKVFloat64Slice(s, got, want, QuantilesTolerance(lower, upper)); err != nil {
			t.Fatalf("ApproxEqualsKVFloat64Slice: got error %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("QuantilesPerKey did apply public partitions correctly: %v", err)
		}
	}
}

func TestQuantilesTolerance(t *testing.T) {
	for _, tc := range []struct {
		minValue      float64
		maxValue      float64
		wantTolerance float64
	}{
		{-5.0, 5.0, 0.00015258789},
		{0.0, 1000.0, 0.01525878906},
	} {
		got := QuantilesTolerance(tc.minValue, tc.maxValue)
		if !cmp.Equal(got, tc.wantTolerance, cmpopts.EquateApprox(0, 1e-9)) { // Allow for floating point arithmetic errors.
			t.Errorf("QuantilesTolerance: with minValue=%f maxValue=%f got tolerance=%f, want=%f", tc.minValue, tc.maxValue, got, tc.wantTolerance)
		}
	}
}

// Tests that SelectPartitions bounds cross-partition contributions correctly and keeps
// all partitions in test mode with PrivatePCollection<V> inputs.
func TestSelectPartitionsTestModeCrossPartitionContributionBoundingV(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		want                     int
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 1,
			// With a max contribution of 1, only 1 partition should be outputted.
			want: 1,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 1,
			// Cross-partition contribution bounding is disabled, so all 10 partitions should be outputted.
			want: 10,
		},
	} {
		// Create 10 partitions with a single privacy ID contributing to each.
		var pairs []testutils.PairII
		for i := 0; i < 10; i++ {
			pairs = append(pairs, testutils.MakePairsWithFixedV(1, i)...)
		}
		p, s, col := ptest.CreateList(pairs)
		col = beam.ParDo(s, testutils.PairToKV, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		got := pbeam.SelectPartitions(s, pcol, pbeam.SelectPartitionsParams{MaxPartitionsContributed: tc.maxPartitionsContributed})

		testutils.CheckNumPartitions(s, got, tc.want)
		if err := ptest.Run(p); err != nil {
			t.Errorf("SelectPartitions: %s did not bound cross-partition contributions correctly for PrivatePCollection<V> inputs: %v", tc.desc, err)
		}
	}
}

// Tests that SelectPartitions bounds cross-partition contributions correctly and keeps
// all partitions in test mode with PrivatePCollection<K,V> inputs.
func TestSelectPartitionsTestModeCrossPartitionContributionBoundingKV(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		want                     int
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 1,
			// With a max contribution of 1, only 1 partition should be outputted.
			want: 1,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 1,
			// Cross-partition contribution bounding is disabled, so all 10 partitions should be outputted.
			want: 10,
		},
	} {
		// Create 10 partitions with a single privacy ID contributing to each.
		var triples []testutils.TripleWithIntValue
		for i := 0; i < 10; i++ {
			triples = append(triples, testutils.MakeTripleWithIntValue(1, i, 0)...)
		}
		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		got := pbeam.SelectPartitions(s, pcol, pbeam.SelectPartitionsParams{MaxPartitionsContributed: tc.maxPartitionsContributed})

		testutils.CheckNumPartitions(s, got, tc.want)
		if err := ptest.Run(p); err != nil {
			t.Errorf("SelectPartitions: %s did not bound cross-partition contributions correctly for PrivatePCollection<K,V> inputs: %v", tc.desc, err)
		}
	}
}

// Tests that DistinctPerKey bounds cross-partition contributions correctly, adds no
// noise and keeps all partitions in test mode.
func TestDistinctPerKeyTestModeCrossPartitionContributionBounding(t *testing.T) {
	for _, tc := range []struct {
		desc                     string
		privacySpec              *pbeam.PrivacySpec
		maxPartitionsContributed int64
		want                     int64
	}{
		{
			desc:                     "test mode with contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes once to 10 partitions, which implies that count of each
			// partition is 1. With a max contribution of 3, 7 partitions should be dropped. The sum of
			// all counts must then be 3. This also ensures that no partitions (each with a single
			// privacy id) gets thresholded.
			want: 3,
		},
		{
			desc:                     "test mode without contribution bounding",
			privacySpec:              NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxPartitionsContributed: 3,
			// The same privacy ID contributes once to 10 partitions, which implies that count of each
			// partition is 3. Cross-partition contribution bounding is disabled. The sum of all counts
			// must then be 10. This also ensures that no partitions (each with a single privacy id)
			// gets thresholded.
			want: 10,
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,1}, {0,1,1}, {0,2,1}, …, {0,9,1}.
		var triples []testutils.TripleWithIntValue
		for i := 0; i < 10; i++ {
			triples = append(triples, testutils.MakeSampleTripleWithIntValue(1, i)...)
		}
		wantMetric := []testutils.TestInt64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		got := pbeam.DistinctPerKey(s, pcol, pbeam.DistinctPerKeyParams{
			MaxPartitionsContributed:     tc.maxPartitionsContributed,
			MaxContributionsPerPartition: 1,
			NoiseKind:                    pbeam.LaplaceNoise{}})
		sums := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, sums)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("DistinctPerKey: %s did not bound cross-partition contributions correctly: %v", tc.desc, err)
		}
	}
}

// Tests that DistinctPerKey bounds per-partition contributions correctly, adds no
// noise and keeps all partitions in test mode.
func TestDistinctPerKeyTestModePerPartitionContributionBounding(t *testing.T) {
	for _, tc := range []struct {
		desc                         string
		privacySpec                  *pbeam.PrivacySpec
		maxContributionsPerPartition int64
		want                         int64
	}{
		{
			desc:                         "test mode with contribution bounding",
			privacySpec:                  NewPrivacySpecNoNoiseWithContributionBounding(tinyEpsilon, tinyDelta),
			maxContributionsPerPartition: 3,
			// MaxContributionsPerPartition = 3, but id = 0 contributes 10 distinct values to partition 0.
			// There will be a per-partition contribution bounding stage and only 3 of 10 distinct values
			// will be kept. The count of partition 0 must then be 3. This also ensures that partition 0
			// (with a single privacy id) does not get thresholded.
			want: 3,
		},
		{
			desc:                         "test mode without contribution bounding",
			privacySpec:                  NewPrivacySpecNoNoiseWithoutContributionBounding(tinyEpsilon, tinyDelta),
			maxContributionsPerPartition: 3,
			// MaxContributionsPerPartition = 3, but id = 0 contributes 10 distinct values to partition 0.
			// There will not be a per-partition contribution bounding stage, so all 10 distinct values will
			// be kept. The count of partition 0 must then be 10. This also ensures that partition 0 (with
			// a single privacy id) does not get thresholded.
			want: 10,
		},
	} {
		// triples{privacy_id, partition_key, value} contains {0,0,0}, {0,0,1}, {0,0,2}, …, {0,0,9}.
		var triples []testutils.TripleWithIntValue
		for i := 0; i < 10; i++ {
			triples = append(triples, testutils.TripleWithIntValue{ID: 0, Partition: 0, Value: i})
		}
		wantMetric := []testutils.TestInt64Metric{
			{0, tc.want},
		}
		p, s, col, want := ptest.CreateList2(triples, wantMetric)
		col = beam.ParDo(s, testutils.ExtractIDFromTripleWithIntValue, col)

		pcol := pbeam.MakePrivate(s, col, tc.privacySpec)
		pcol = pbeam.ParDo(s, testutils.TripleWithIntValueToKV, pcol)
		got := pbeam.DistinctPerKey(s, pcol, pbeam.DistinctPerKeyParams{
			MaxPartitionsContributed:     1,
			MaxContributionsPerPartition: tc.maxContributionsPerPartition,
			NoiseKind:                    pbeam.LaplaceNoise{}})
		sums := beam.DropKey(s, got)
		sumOverPartitions := stats.Sum(s, sums)
		got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
		want = beam.ParDo(s, testutils.Int64MetricToKV, want)
		if err := testutils.EqualsKVInt64(s, got, want); err != nil {
			t.Fatalf("EqualsKVInt64: %v", err)
		}
		if err := ptest.Run(p); err != nil {
			t.Errorf("DistinctPerKey: %s did not bound per-partition contributions correctly: %v", tc.desc, err)
		}
	}
}
