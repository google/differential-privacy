package pbeam

import (
	"testing"

	"github.com/google/differential-privacy/go/dpagg"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/testing/ptest"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
)

// Checks that DistinctPrivacyKey returns a correct answer, in particular that values
// are correctly counted (without duplicates).
func TestDistinctPrivacyKeyNoNoise(t *testing.T) {
	var triples []tripleWithIntValue
	for i := 0; i < 100; i++ { // Add 400 values of which 200 are distinct to Partition 0.
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: 100 + i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: i})       // Duplicate each value. Should be discarded by DistinctPerKey.
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: 100 + i}) // Duplicate each value. Should be discarded by DistinctPerKey.
	}
	for i := 0; i < 50; i++ { // Add 200 values of which 100 are distinct to Partition 1.
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 1, Value: i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 1, Value: 50 + i})
		// Have 2 users contribute to the same 100 distinct values.
		triples = append(triples, tripleWithIntValue{ID: 100 + i, Partition: 1, Value: i})      // Should be discarded.
		triples = append(triples, tripleWithIntValue{ID: 100 + i, Partition: 1, Value: 50 + i}) // Should be discarded.
	}
	for i := 0; i < 7; i++ { // Add 7 distinct values to Partition 2. Should be thresholded.
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 2, Value: i})
	}
	result := []testInt64Metric{
		{0, 200},
		{1, 100},
		// Only 7 distinct values in partition 2: should be thresholded.
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=50, δ=10⁻¹⁰⁰ and l1Sensitivity=6 gives a threshold of ≈33.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 1-10⁻²⁵ probability (k=25).
	// To see the logic and the math behind flakiness and tolerance calculation,
	// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-100, 25.0, 6.0
	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := DistinctPerKey(s, pcol, DistinctPerKeyParams{MaxPartitionsContributed: 3, NoiseKind: LaplaceNoise{}, MaxContributionsPerPartition: 2})
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestDistinctPerKeyNoNoise: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPerKeyNoNoise: DistinctPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

// Checks that DistinctPrivacyKey adds noise to its output. The logic mirrors TestDistinctPrivacyIDAddsNoise.
func TestDistinctPrivacyKeyAddsNoise(t *testing.T) {
	for _, tc := range []struct {
		name      string
		noiseKind NoiseKind
		// Differential privacy params used.
		epsilon float64
		delta   float64
	}{
		{
			name:      "Gaussian",
			noiseKind: GaussianNoise{},
			epsilon:   2 * 1e-15, // It is split by 2: 1e-15 for the noise and 1e-15 for the partition selection.
			delta:     2 * 1e-5,  // It is split by 2: 1e-5 for the noise and 1e-5 for the partition selection.
		},
		{
			name:      "Laplace",
			noiseKind: LaplaceNoise{},
			epsilon:   2 * 1e-15, // It is split by 2: 1e-15 for the noise and 1e-15 for the partition selection.
			delta:     0.01,
		},
	} {
		// Because this is an integer aggregation, we can't use the regular complementary
		// tolerance computations. Instead, we do the following:
		//
		// If generated noise is between -0.5 and 0.5, it will be rounded to 0 and the
		// test will fail. For Laplace, this will happen with probability
		//   P ~= Laplace_CDF(0.5) - Laplace_CDF(-0.5).
		// Given that Laplace scale = l1_sensitivity / ε = 10¹⁵,, P ~= 5e-16.
		// For Gaussian, this will happen with probability
		//	 P ~= Gaussian_CDF(0.5) - Gaussian_CDF(-0.5).
		// For given ε=1e-15, δ=1e-5 => sigma = 39904, P ~= 1e-5.
		//
		// We want to keep numIDs low (otherwise the tests take a long time) while
		// also keeping P low. We use magic partition selection here, meaning that
		// numIDs cap at 1/δ. So, we can have tiny epsilon without having to worry
		// about tests taking long.
		tolerance := 0.0
		l0Sensitivity, lInfSensitivity := int64(1), int64(1)
		partitionSelectionEpsilon, partitionSelectionDelta := tc.epsilon/2, tc.delta
		if tc.noiseKind == gaussianNoise {
			partitionSelectionDelta = tc.delta / 2
		}

		// Compute the number of IDs needed to keep the partition.
		sp := dpagg.NewPreAggSelectPartition(
			&dpagg.PreAggSelectPartitionOptions{
				Epsilon:                  partitionSelectionEpsilon,
				Delta:                    partitionSelectionDelta,
				MaxPartitionsContributed: l0Sensitivity,
			})
		numIDs := sp.GetHardThreshold()

		triples := make([]tripleWithIntValue, numIDs)
		for i := 0; i < numIDs; i++ { // Add numIDs distinct values to Partition 0.
			triples[i] = tripleWithIntValue{ID: i, Partition: 0, Value: i}
		}
		p, s, col := ptest.CreateList(triples)
		col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

		pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
		pcol = ParDo(s, tripleWithIntValueToKV, pcol)
		got := DistinctPerKey(s, pcol, DistinctPerKeyParams{MaxPartitionsContributed: int64(l0Sensitivity), NoiseKind: LaplaceNoise{}, MaxContributionsPerPartition: int64(lInfSensitivity)})
		got = beam.ParDo(s, kvToInt64Metric, got)

		checkInt64MetricsAreNoisy(s, got, numIDs, tolerance)
		if err := ptest.Run(p); err != nil {
			t.Errorf("DistinctPerKey didn't add any %s noise: %v", tc.name, err)
		}
	}
}

// Checks that DistinctPerKey bounds cross-partition contributions correctly.
// The logic mirrors TestCountCrossPartitionContributionBounding.
func TestDistinctPerKeyPerKeyCrossPartitionContributionBounding(t *testing.T) {
	var triples []tripleWithIntValue
	for i := 0; i < 10; i++ { // Add 10 partitions.
		for j := 0; j < 50; j++ { // Add 50 distinct values to each partition.
			triples = append(triples, tripleWithIntValue{ID: j, Partition: i, Value: j})
			triples = append(triples, tripleWithIntValue{ID: j, Partition: i, Value: j}) // Duplicate each value. Should be discarded.
		}
	}
	result := []testInt64Metric{
		{0, 150},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=50, δ=0.01 and l1Sensitivity=3 gives post-aggregation threshold of 1.
	// We have 10 partitions. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²⁵ probability (k=25).
	epsilon, delta, k, l1Sensitivity := 50.0, 0.01, 25.0, 3.0
	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := DistinctPerKey(s, pcol, DistinctPerKeyParams{MaxPartitionsContributed: 3, NoiseKind: LaplaceNoise{}, MaxContributionsPerPartition: 1})
	// With a max contribution of 3, 70% of the data should have be
	// dropped. The sum of all elements must then be 150.
	counts := beam.DropKey(s, got)
	sumOverPartitions := stats.Sum(s, counts)
	got = beam.AddFixedKey(s, sumOverPartitions) // Adds a fixed key of 0.
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestDistinctPerKeyCrossPartitionContributionBounding: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPerKeyCrossPartitionContributionBounding: DistinctPerKey(%v) = %v, expected elements to sum to 150: %v", col, got, err)
	}
}

// Checks that DistinctPrivacyKey bounds per-partition contributions correctly.
func TestDistinctPrivacyKeyPerPartitionContributionBounding(t *testing.T) {
	var triples []tripleWithIntValue
	for i := 0; i < 100; i++ { // Add 500 distinct values to Partition 0.
		// MaxContributionsPerPartition is set to 2, so 3 of these 5 contributions will be dropped for each user.
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: 100 + i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: 200 + i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: 300 + i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: 400 + i})
	}
	for i := 0; i < 50; i++ { // Add 200 distinct values to Partition 1.
		// MaxContributionsPerPartition is set to 2, so 2 of these 4 contributions will be dropped for each user.
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 1, Value: i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 1, Value: 50 + i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 1, Value: 100 + i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 1, Value: 150 + i})
	}
	for i := 0; i < 50; i++ { // Add 150 distinct values to Partition 2.
		// MaxContributionsPerPartition is set to 2, so 1 of these 3 contributions will be dropped for each user.
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 2, Value: i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 2, Value: 50 + i})
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 2, Value: 100 + i})
	}
	result := []testInt64Metric{
		{0, 200}, // 300 distinct values will be dropped due to per-partition contribution bounding.
		{1, 100}, // 100 distinct values will be dropped due to per-partition contribution bounding.
		{2, 100}, // 50 distinct values will be dropped due to per-partition contribution bounding.
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=50, δ=10⁻¹⁰⁰ and l1Sensitivity=6 gives a threshold of ≈33.
	// We have 3 partitions. So, to get an overall flakiness of 10⁻²³,
	// we can have each partition fail with 1-10⁻²⁵ probability (k=25).
	// To see the logic and the math behind flakiness and tolerance calculation,
	// See https://github.com/google/differential-privacy/blob/main/privacy-on-beam/docs/Tolerance_Calculation.pdf.
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-100, 25.0, 6.0
	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := DistinctPerKey(s, pcol, DistinctPerKeyParams{MaxPartitionsContributed: 3, NoiseKind: LaplaceNoise{}, MaxContributionsPerPartition: 2})
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestDistinctPerKeyNoNoise: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPerKeyNoNoise: DistinctPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}

var distinctPerKeyPartitionSelectionNonDeterministicTestCases = []struct {
	name                string
	noiseKind           NoiseKind
	epsilon             float64
	delta               float64
	numPartitions       int
	entriesPerPartition int
}{
	{
		name:      "Gaussian",
		noiseKind: GaussianNoise{},
		// After splitting the (ε, δ) budget between the noise and partition
		// selection portions of the privacy algorithm, this results in a ε=1,
		// δ=0.3 partition selection budget.
		epsilon: 2,
		delta:   0.6,
		// entriesPerPartition=1 yields a 30% chance of emitting any particular partition
		// (since δ_emit=0.3).
		entriesPerPartition: 1,
		// 143 distinct partitions implies that some (but not all) partitions are
		// emitted with high probability (at least 1 - 1e-20).
		numPartitions: 143,
	},
	{
		name:      "Laplace",
		noiseKind: LaplaceNoise{},
		// After splitting the (ε, δ) budget between the noise and partition
		// selection portions of the privacy algorithm, this results in the
		// partition selection portion of the budget being ε_selectPartition=1,
		// δ_selectPartition=0.3.
		epsilon: 2,
		delta:   0.3,
		// entriesPerPartition=1 yields a 30% chance of emitting any particular partition
		// (since δ_emit=0.3).
		entriesPerPartition: 1,
		numPartitions:       143,
	},
}

// Checks that DistinctPerKey is performing a random partition selection.
func TestDistinctPerKeyPartitionSelectionNonDeterministicInt(t *testing.T) {
	for _, tc := range distinctPerKeyPartitionSelectionNonDeterministicTestCases {
		t.Run(tc.name, func(t *testing.T) {
			// Sanity check that the entriesPerPartition is sensical.
			if tc.entriesPerPartition <= 0 {
				t.Fatalf("Invalid test case: entriesPerPartition must be positive. Got: %d", tc.entriesPerPartition)
			}

			// Build up {ID, Partition, Value} pairs such that for each of the tc.numPartitions partitions,
			// tc.entriesPerPartition users contribute a single value:
			//    {0, 0, 1}, {1, 0, 1}, …, {entriesPerPartition-1, 0, 1}
			//    {entriesPerPartition, 1, 1}, {entriesPerPartition+1, 1, 1}, …, {entriesPerPartition+entriesPerPartition-1, 1, 1}
			//    …
			//    {entriesPerPartition*(numPartitions-1), numPartitions-1, 1}, …, {entriesPerPartition*numPartitions-1, numPartitions-1, 1}
			var (
				triples []tripleWithIntValue
				kOffset = 0
			)
			for i := 0; i < tc.numPartitions; i++ {
				for j := 0; j < tc.entriesPerPartition; j++ {
					triples = append(triples, tripleWithIntValue{ID: kOffset + j, Partition: i, Value: 1})
				}
				kOffset += tc.entriesPerPartition
			}
			p, s, col := ptest.CreateList(triples)
			col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

			// Run DistinctPerKey on triples
			pcol := MakePrivate(s, col, NewPrivacySpec(tc.epsilon, tc.delta))
			pcol = ParDo(s, tripleWithIntValueToKV, pcol)
			got := DistinctPerKey(s, pcol, DistinctPerKeyParams{MaxPartitionsContributed: int64(tc.numPartitions), NoiseKind: tc.noiseKind, MaxContributionsPerPartition: 1})
			got = beam.ParDo(s, kvToInt64Metric, got)

			// Validate that partitions are selected randomly (i.e., some emitted and some dropped).
			checkSomePartitionsAreDropped(s, got, tc.numPartitions)
			if err := ptest.Run(p); err != nil {
				t.Errorf("%v", err)
			}
		})
	}
}

// Checks that DistinctPrivacyKey performs thresholding/partition selection
// on the number of privacy IDs in a partition and not the number of distinct
// values.
func TestDistinctPerKeyThresholdsOnPrivacyIDs(t *testing.T) {
	var triples []tripleWithIntValue
	for i := 0; i < 10; i++ { // Add 10 users (each contributing the same value) to Partition 1.
		triples = append(triples, tripleWithIntValue{ID: i, Partition: 0, Value: 0})
	}
	result := []testInt64Metric{
		{0, 1},
	}
	p, s, col, want := ptest.CreateList2(triples, result)
	col = beam.ParDo(s, extractIDFromTripleWithIntValue, col)

	// ε=50, δ=10⁻¹⁰⁰ and l1Sensitivity=1 gives a threshold of ≈6.
	// We have 1 partition. So, to get an overall flakiness of 10⁻²³,
	// we need to have each partition pass with 1-10⁻²³ probability (k=23).
	epsilon, delta, k, l1Sensitivity := 50.0, 1e-100, 23.0, 1.0
	// ε is split by 2 for noise and for partition selection, so we use 2*ε to get a Laplace noise with ε.
	pcol := MakePrivate(s, col, NewPrivacySpec(2*epsilon, delta))
	pcol = ParDo(s, tripleWithIntValueToKV, pcol)
	got := DistinctPerKey(s, pcol, DistinctPerKeyParams{MaxPartitionsContributed: 1, NoiseKind: LaplaceNoise{}, MaxContributionsPerPartition: 1})
	want = beam.ParDo(s, int64MetricToKV, want)
	if err := approxEqualsKVInt64(s, got, want, laplaceTolerance(k, l1Sensitivity, epsilon)); err != nil {
		t.Fatalf("TestDistinctPerKeyNoNoise: %v", err)
	}
	if err := ptest.Run(p); err != nil {
		t.Errorf("TestDistinctPerKeyNoNoise: DistinctPerKey(%v) = %v, expected %v: %v", col, got, want, err)
	}
}
