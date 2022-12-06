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

package pbeam

import (
	"bytes"
	"fmt"
	"reflect"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/v2/checks"
	"github.com/google/differential-privacy/go/v2/dpagg"
	"github.com/google/differential-privacy/go/v2/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v2/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/transforms/stats"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*prepareSumFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*addNoiseToEmptyPublicPartitionsInt64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*addNoiseToEmptyPublicPartitionsFloat64Fn)(nil)))
	// TODO: add tests to make sure we don't forget anything here
}

// SumParams specifies the parameters associated with a Sum aggregation.
type SumParams struct {
	// Noise type (which is either LaplaceNoise{} or GaussianNoise{}).
	//
	// Defaults to LaplaceNoise{}.
	NoiseKind NoiseKind
	// Differential privacy budget consumed by this aggregation. If there is
	// only one aggregation, both Epsilon and Delta can be left 0; in that
	// case, the entire budget of the PrivacySpec is consumed.
	Epsilon, Delta float64
	// The maximum number of distinct values that a given privacy identifier
	// can influence. There is an inherent trade-off when choosing this
	// parameter: a larger MaxPartitionsContributed leads to less data loss due
	// to contribution bounding, but since the noise added in aggregations is
	// scaled according to maxPartitionsContributed, it also means that more
	// noise is added to each count.
	//
	// Required.
	MaxPartitionsContributed int64
	// The total contribution of a given privacy identifier to partition can be
	// at at least MinValue, and at most MaxValue; otherwise it will be clamped
	// to these bounds. For example, if a privacy identifier is associated with
	// the key-value pairs [("a", -5), ("a", 2), ("b", 7), ("c", 3)] and the
	// (MinValue, MaxValue) bounds are (0, 5), the contribution for "a" will be
	// clamped up to 0, the contribution for "b" will be clamped down to 5, and
	// the contribution for "c" will be untouched. There is an inherent
	// trade-off when choosing MinValue and MaxValue: a small MinValue and a
	// large MaxValue means that less records will be clamped, but that more
	// noise will be added.
	//
	// Required.
	MinValue, MaxValue float64
	// You can input the list of partitions present in the output if you know
	// them in advance. When you specify partitions, partition selection /
	// thresholding will be disabled and partitions will appear in the output
	// if and only if they appear in the set of public partitions.
	//
	// You should not derive the list of partitions non-privately from private
	// data. You should only use this in either of the following cases:
	// 	1. The list of partitions is data-independent. For example, if you are
	// 	aggregating a metric by hour, you could provide a list of all possible
	// 	hourly period.
	// 	2. You use a differentially private operation to come up with the list of
	// 	partitions. For example, you could use the output of a SelectPartitions
	//  operation or the keys of a DistinctPrivacyID operation as the list of
	//  public partitions.
	//
	// PublicPartitions needs to be a beam.PCollection, slice, or array. The
	// underlying type needs to match the partition type of the PrivatePCollection.
	//
	// Prefer slices or arrays if the list of public partitions is small and
	// can fit into memory (e.g., up to a million). Prefer beam.PCollection
	// otherwise.
	//
	// Optional.
	PublicPartitions any
}

// SumPerKey sums the values associated with each key in a
// PrivatePCollection<K,V>, adding differentially private noise to the sums and
// doing pre-aggregation thresholding to remove sums with a low number of
// distinct privacy identifiers. Client can also specify a PCollection of partitions.
//
// It is also possible to manually specify the list of partitions
// present in the output, in which case the partition selection/thresholding
// step is skipped.
//
// SumPerKey transforms a PrivatePCollection<K,V> either into a
// PCollection<K,int64> or a PCollection<K,float64>, depending on whether its
// input is an integer type or a float type.
//
// Note: Do not use when your results may cause overflows for int64 and float64
// values. This aggregation is not hardened for such applications yet.
func SumPerKey(s beam.Scope, pcol PrivatePCollection, params SumParams) beam.PCollection {
	s = s.Scope("pbeam.SumPerKey")
	// Obtain & validate type information from the underlying PCollection<K,V>.
	idT, kvT := beam.ValidateKVType(pcol.col)
	if kvT.Type() != reflect.TypeOf(kv.Pair{}) {
		log.Fatalf("SumPerKey must be used on a PrivatePCollection of type <K,V>, got type %v instead", kvT)
	}
	if pcol.codec == nil {
		log.Fatalf("SumPerKey: no codec found for the input PrivatePCollection.")
	}

	// Get privacy parameters.
	spec := pcol.privacySpec
	epsilon, delta, err := spec.consumeBudget(params.Epsilon, params.Delta)
	if err != nil {
		log.Fatalf("Couldn't consume budget for SumPerKey: %v", err)
	}

	var noiseKind noise.Kind
	if params.NoiseKind == nil {
		noiseKind = noise.LaplaceNoise
		log.Infof("No NoiseKind specified, using Laplace Noise by default.")
	} else {
		noiseKind = params.NoiseKind.toNoiseKind()
	}
	err = checkSumPerKeyParams(params, epsilon, delta, noiseKind, pcol.codec.KType.T)
	if err != nil {
		log.Fatalf("pbeam.SumPerKey: %v", err)
	}

	maxPartitionsContributed, err := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	if err != nil {
		log.Fatalf("Couldn't get MaxPartitionsContributed for SumPerKey: %v", err)
	}
	// Drop non-public partitions, if public partitions are specified.
	pcol.col, err = dropNonPublicPartitions(s, pcol, params.PublicPartitions, pcol.codec.KType.T)
	if err != nil {
		log.Fatalf("Couldn't drop non-public partitions for SumPerKey: %v", err)
	}
	// First, group together the privacy ID and the partition ID, and sum the
	// values per-privacy unit and per-partition.
	decoded := beam.ParDo(s,
		newPrepareSumFn(idT, pcol.codec),
		pcol.col,
		beam.TypeDefinition{Var: beam.VType, T: pcol.codec.VType.T})
	summed := stats.SumPerKey(s, decoded)
	// Second, convert the sum to int64 or float64, and re-key.
	_, sumT := beam.ValidateKVType(summed)
	convertFn, err := findConvertFn(sumT)
	if err != nil {
		log.Fatalf("Couldn't get convertFn for SumPerKey: %v", err)
	}
	vKind, err := getKind(convertFn)
	if err != nil {
		log.Fatalf("Couldn't get vKind for SumPerKey: %v", err)
	}
	converted := beam.ParDo(s, convertFn, summed)
	rekeyFn, err := findRekeyFn(vKind)
	if err != nil {
		log.Fatalf("Couldn't get rekeyFn for SumPerKey: %v", err)
	}
	rekeyed := beam.ParDo(s, rekeyFn, converted)
	// Second, do cross-partition contribution bounding if not in test mode without contribution bounding.
	if spec.testMode != noNoiseWithoutContributionBounding {
		rekeyed = boundContributions(s, rekeyed, maxPartitionsContributed)
	}
	// Fourth, now that contribution bounding is done, remove the privacy keys,
	// decode the value, and do a DP sum with all the partial sums.
	partialSumPairs := beam.DropKey(s, rekeyed)
	partitionT := pcol.codec.KType.T
	decodePairFn, err := newDecodePairFn(partitionT, vKind)
	if err != nil {
		log.Fatalf("Couldn't get decodePairFn for SumPerKey: %v", err)
	}
	partialSumKV := beam.ParDo(s,
		decodePairFn,
		partialSumPairs,
		beam.TypeDefinition{Var: beam.XType, T: partitionT})

	var result beam.PCollection
	// Add public partitions and return the aggregation output, if public partitions are specified.
	if params.PublicPartitions != nil {
		result = addPublicPartitionsForSum(s, epsilon, delta, maxPartitionsContributed,
			params, noiseKind, vKind, partialSumKV, spec.testMode)
	} else {
		boundedSumFn, err := newBoundedSumFn(epsilon, delta, maxPartitionsContributed, params.MinValue, params.MaxValue, noiseKind, vKind, false, spec.testMode)
		if err != nil {
			log.Fatalf("Couldn't get boundedSumFn for SumPerKey: %v", err)
		}
		sums := beam.CombinePerKey(s,
			boundedSumFn,
			partialSumKV)
		// Drop thresholded partitions.
		dropThresholdedPartitionsFn, err := findDropThresholdedPartitionsFn(vKind)
		if err != nil {
			log.Fatalf("Couldn't get dropThresholdedPartitionsFn for SumPerKey: %v", err)
		}
		result = beam.ParDo(s, dropThresholdedPartitionsFn, sums)
	}

	// Clamp negative counts to zero when MinValue is non-negative.
	if params.MinValue >= 0 {
		clampNegativePartitionsFn, err := findClampNegativePartitionsFn(vKind)
		if err != nil {
			log.Fatalf("Couldn't get clampNegativePartitionsFn for SumPerKey: %v", err)
		}
		result = beam.ParDo(s, clampNegativePartitionsFn, result)
	}

	return result
}

func addPublicPartitionsForSum(s beam.Scope, epsilon, delta float64, maxPartitionsContributed int64, params SumParams, noiseKind noise.Kind, vKind reflect.Kind, partialSumKV beam.PCollection, testMode testMode) beam.PCollection {
	// Calculate sums with non-public partitions dropped. Result is PCollection<partition, int64> or PCollection<partition, float64>.
	boundedSumFn, err := newBoundedSumFn(epsilon, delta, maxPartitionsContributed, params.MinValue, params.MaxValue, noiseKind, vKind, true, testMode)
	if err != nil {
		log.Fatalf("Couldn't get boundedSumFn for SumPerKey: %v", err)
	}
	sums := beam.CombinePerKey(s,
		boundedSumFn,
		partialSumKV)
	partitionT, _ := beam.ValidateKVType(sums)
	sumsPartitions := beam.DropValue(s, sums)
	// Create map with partitions in the data as keys.
	partitionMap := beam.Combine(s, newPartitionsMapFn(beam.EncodedType{partitionT.Type()}), sumsPartitions)
	// Add value of 0 to each partition key in PublicPartitions.
	addZeroValuesToPublicPartitions, err := newAddZeroValuesToPublicPartitionsFn(vKind)
	if err != nil {
		log.Fatalf("Couldn't get addZeroValuesToPublicPartitions for SumPerKey: %v", err)
	}
	publicPartitions, isPCollection := params.PublicPartitions.(beam.PCollection)
	if !isPCollection {
		publicPartitions = beam.Reshuffle(s, beam.CreateList(s, params.PublicPartitions))
	}
	publicPartitionsWithValues := beam.ParDo(s, addZeroValuesToPublicPartitions, publicPartitions)
	// emptyPublicPartitions are the partitions that are public but not found in the data.
	emptyPublicPartitions := beam.ParDo(s, newEmitPartitionsNotInTheDataFn(partitionT), publicPartitionsWithValues, beam.SideInput{Input: partitionMap})
	// Add noise to the empty public partitions.
	addNoiseToEmptyPublicPartitionsFn, err := newAddNoiseToEmptyPublicPartitionsFn(epsilon, delta, maxPartitionsContributed, params.MinValue, params.MaxValue, noiseKind, vKind, testMode)
	if err != nil {
		log.Fatalf("Couldn't get addNoiseToEmptyPublicPartitionsFn for SumPerKey: %v", err)
	}
	emptySums := beam.ParDo(s,
		addNoiseToEmptyPublicPartitionsFn,
		emptyPublicPartitions)
	dereferenceValueFn, err := findDereferenceValueFn(vKind)
	if err != nil {
		log.Fatalf("Couldn't get dereferenceValueFn for SumPerKey: %v", err)
	}
	sums = beam.ParDo(s, dereferenceValueFn, sums)
	// Merge sums from data with sums from the empty public partitions and return.
	return beam.Flatten(s, sums, emptySums)
}

func checkSumPerKeyParams(params SumParams, epsilon, delta float64, noiseKind noise.Kind, partitionType reflect.Type) error {
	err := checkPublicPartitions(params.PublicPartitions, partitionType)
	if err != nil {
		return err
	}
	err = checks.CheckEpsilon(epsilon)
	if err != nil {
		return err
	}
	err = checkDelta(delta, noiseKind, params.PublicPartitions)
	if err != nil {
		return err
	}
	return checks.CheckBoundsFloat64(params.MinValue, params.MaxValue)
}

// prepareSumFn takes a PCollection<ID,kv.Pair{K,V}> as input, and returns a
// PCollection<kv.Pair{ID,K},V>; where ID has been coded, and V has been
// decoded.
type prepareSumFn struct {
	IDType         beam.EncodedType
	idEnc          beam.ElementEncoder
	InputPairCodec *kv.Codec
}

func newPrepareSumFn(idType typex.FullType, kvCodec *kv.Codec) *prepareSumFn {
	return &prepareSumFn{
		IDType:         beam.EncodedType{idType.Type()},
		InputPairCodec: kvCodec,
	}
}

func (fn *prepareSumFn) Setup() error {
	fn.idEnc = beam.NewElementEncoder(fn.IDType.T)
	return fn.InputPairCodec.Setup()
}

func (fn *prepareSumFn) ProcessElement(id beam.W, pair kv.Pair) (kv.Pair, beam.V, error) {
	var idBuf bytes.Buffer
	if err := fn.idEnc.Encode(id, &idBuf); err != nil {
		return kv.Pair{}, nil, fmt.Errorf("pbeam.prepareSumFn.ProcessElement: couldn't encode ID %v: %w", id, err)
	}
	_, v, err := fn.InputPairCodec.Decode(pair)
	return kv.Pair{idBuf.Bytes(), pair.K}, v, err
}

// findConvertFn gets the correct conversion to int64 or float64 function.
func findConvertFn(t typex.FullType) (any, error) {
	switch t.Type().String() {
	case "int":
		return convertIntToInt64Fn, nil
	case "int8":
		return convertInt8ToInt64Fn, nil
	case "int16":
		return convertInt16ToInt64Fn, nil
	case "int32":
		return convertInt32ToInt64Fn, nil
	case "int64":
		return convertInt64ToInt64Fn, nil
	case "uint":
		return convertUintToInt64Fn, nil
	case "uint8":
		return convertUint8ToInt64Fn, nil
	case "uint16":
		return convertUint16ToInt64Fn, nil
	case "uint32":
		return convertUint32ToInt64Fn, nil
	case "uint64":
		return convertUint64ToInt64Fn, nil
	case "float32":
		return convertFloat32ToFloat64Fn, nil
	case "float64":
		return convertFloat64ToFloat64Fn, nil
	default:
		return nil, fmt.Errorf("unexpected value type of %v", t)
	}
}

func convertIntToInt64Fn(z beam.Z, i int) (beam.Z, int64) {
	return z, int64(i)
}
func convertInt8ToInt64Fn(z beam.Z, i int8) (beam.Z, int64) {
	return z, int64(i)
}
func convertInt16ToInt64Fn(z beam.Z, i int16) (beam.Z, int64) {
	return z, int64(i)
}
func convertInt32ToInt64Fn(z beam.Z, i int32) (beam.Z, int64) {
	return z, int64(i)
}
func convertInt64ToInt64Fn(z beam.Z, i int64) (beam.Z, int64) {
	return z, i
}
func convertUintToInt64Fn(z beam.Z, i uint) (beam.Z, int64) {
	return z, int64(i)
}
func convertUint8ToInt64Fn(z beam.Z, i uint8) (beam.Z, int64) {
	return z, int64(i)
}
func convertUint16ToInt64Fn(z beam.Z, i uint16) (beam.Z, int64) {
	return z, int64(i)
}
func convertUint32ToInt64Fn(z beam.Z, i uint32) (beam.Z, int64) {
	return z, int64(i)
}
func convertUint64ToInt64Fn(z beam.Z, i uint64) (beam.Z, int64) {
	return z, int64(i)
}

// getKind gets the return kind of the convertFn function.
func getKind(fn any) (reflect.Kind, error) {
	if fn == nil {
		return reflect.Invalid, fmt.Errorf("convertFn is nil")
	}
	if reflect.TypeOf(fn).Kind() != reflect.Func {
		return reflect.Invalid, fmt.Errorf("convertFn is %v, should be a function", reflect.TypeOf(fn).Kind())
	}
	if reflect.TypeOf(fn).NumOut() < 2 {
		return reflect.Invalid, fmt.Errorf("convertFn has %v outputs, expected at least 2", reflect.TypeOf(fn).NumOut())
	}
	return reflect.TypeOf(fn).Out(1).Kind(), nil
}

func newAddNoiseToEmptyPublicPartitionsFn(epsilon, delta float64, maxPartitionsContributed int64, lower, upper float64, noiseKind noise.Kind, vKind reflect.Kind, testMode testMode) (any, error) {
	var err error
	var bsFn any

	switch vKind {
	case reflect.Int64:
		err = checks.CheckBoundsFloat64AsInt64(lower, upper)
		bsFn = newAddNoiseToEmptyPublicPartitionsInt64Fn(epsilon, delta, maxPartitionsContributed, int64(lower), int64(upper), noiseKind, testMode)
	case reflect.Float64:
		err = checks.CheckBoundsFloat64(lower, upper)
		bsFn = newAddNoiseToEmptyPublicPartitionsFloat64Fn(epsilon, delta, maxPartitionsContributed, lower, upper, noiseKind, testMode)
	default:
		err = fmt.Errorf("vKind(%v) should be int64 or float64", vKind)
	}

	return bsFn, err
}

// addNoiseToEmptyPublicPartitionsInt64Fn adds integer noise to empty partitions.
type addNoiseToEmptyPublicPartitionsInt64Fn struct {
	// Privacy spec parameters (set during initial construction).
	NoiseEpsilon             float64
	NoiseDelta               float64
	MaxPartitionsContributed int64
	Lower                    int64
	Upper                    int64
	NoiseKind                noise.Kind
	noise                    noise.Noise // Set during Setup phase according to NoiseKind.
	TestMode                 testMode
}

// newAddNoiseToEmptyPublicPartitionsInt64Fn returns a addNoiseToEmptyPublicPartitionsInt64Fn with the given budget and parameters.
func newAddNoiseToEmptyPublicPartitionsInt64Fn(epsilon, delta float64, maxPartitionsContributed, lower, upper int64, noiseKind noise.Kind, testMode testMode) *addNoiseToEmptyPublicPartitionsInt64Fn {
	return &addNoiseToEmptyPublicPartitionsInt64Fn{
		NoiseEpsilon:             epsilon,
		NoiseDelta:               delta,
		MaxPartitionsContributed: maxPartitionsContributed,
		Lower:                    lower,
		Upper:                    upper,
		NoiseKind:                noiseKind,
		TestMode:                 testMode,
	}
}

func (fn *addNoiseToEmptyPublicPartitionsInt64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
	if fn.TestMode.isEnabled() {
		fn.noise = noNoise{}
	}
}

func (fn *addNoiseToEmptyPublicPartitionsInt64Fn) ProcessElement(partitionKey beam.X, _ int64) (beam.X, int64, error) {
	bs, err := dpagg.NewBoundedSumInt64(&dpagg.BoundedSumInt64Options{
		Epsilon:                  fn.NoiseEpsilon,
		Delta:                    fn.NoiseDelta,
		MaxPartitionsContributed: fn.MaxPartitionsContributed,
		Lower:                    fn.Lower,
		Upper:                    fn.Upper,
		Noise:                    fn.noise,
	})
	if err != nil {
		return partitionKey, 0, err
	}
	noisedValue, err := bs.Result()
	return partitionKey, noisedValue, err
}

// addNoiseToEmptyPublicPartitionsFloat64Fn adds integer noise to empty partitions.
type addNoiseToEmptyPublicPartitionsFloat64Fn struct {
	// Privacy spec parameters (set during initial construction).
	NoiseEpsilon             float64
	NoiseDelta               float64
	MaxPartitionsContributed int64
	Lower                    float64
	Upper                    float64
	NoiseKind                noise.Kind
	noise                    noise.Noise // Set during Setup phase according to NoiseKind.
	TestMode                 testMode
}

// newAddNoiseToEmptyPublicPartitionsFloat64Fn returns a addNoiseToEmptyPublicPartitionsFloat64Fn with the given budget and parameters.
func newAddNoiseToEmptyPublicPartitionsFloat64Fn(epsilon, delta float64, maxPartitionsContributed int64, lower, upper float64, noiseKind noise.Kind, testMode testMode) *addNoiseToEmptyPublicPartitionsFloat64Fn {
	return &addNoiseToEmptyPublicPartitionsFloat64Fn{
		NoiseEpsilon:             epsilon,
		NoiseDelta:               delta,
		MaxPartitionsContributed: maxPartitionsContributed,
		Lower:                    lower,
		Upper:                    upper,
		NoiseKind:                noiseKind,
		TestMode:                 testMode,
	}
}

func (fn *addNoiseToEmptyPublicPartitionsFloat64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
	if fn.TestMode.isEnabled() {
		fn.noise = noNoise{}
	}
}

func (fn *addNoiseToEmptyPublicPartitionsFloat64Fn) ProcessElement(partitionKey beam.X, _ float64) (beam.X, float64, error) {
	bs, err := dpagg.NewBoundedSumFloat64(&dpagg.BoundedSumFloat64Options{
		Epsilon:                  fn.NoiseEpsilon,
		Delta:                    fn.NoiseDelta,
		MaxPartitionsContributed: fn.MaxPartitionsContributed,
		Lower:                    fn.Lower,
		Upper:                    fn.Upper,
		Noise:                    fn.noise,
	})
	if err != nil {
		return partitionKey, 0, err
	}
	noisedValue, err := bs.Result()
	return partitionKey, noisedValue, err
}
