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
	"github.com/google/differential-privacy/go/checks"
	"github.com/google/differential-privacy/go/dpagg"
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/core/typex"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*boundedMeanFloat64Fn)(nil)))
	beam.RegisterType(reflect.TypeOf((*prepareMeanFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*expandValuesCombineFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*decodePairArrayFloat64Fn)(nil)))
}

// MeanParams specifies the parameters associated with a Mean aggregation.
type MeanParams struct {
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
	// noise is added to each mean.
	//
	// Required.
	MaxPartitionsContributed int64
	// The maximum number of contribution from a given privacy identifier
	// There is an inherent trade-off when choosing this
	// parameter: a larger MaxContributionsPerPartition leads to less data loss due
	// to contribution bounding, but since the noise added in aggregations is
	// scaled according to maxContributionsPerPartition, it also means that more
	// noise is added to each mean.
	//
	// Required.
	MaxContributionsPerPartition int64
	// The total contribution of a given privacy identifier to partition can be
	// at at least MinValue, and at most MaxValue; otherwise it will be clamped
	// to these bounds. For example, if a privacy identifier is associated to
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
}

// MeanPerKey obtains the mean of the values associated with each key in a
// PrivatePCollection<K,V>, adding differentially private noise to the means and
// doing pre-aggregation thresholding to remove means with a low number of
// distinct privacy identifiers.
//
// Note: Do not use when your results may cause overflows for Int64 or Float64
// values.  This aggregation is not hardened for such applications yet.
//
// MeanPerKey transforms a PrivatePCollection<K,V> into a PCollection<K,float64>.
func MeanPerKey(s beam.Scope, pcol PrivatePCollection, params MeanParams) beam.PCollection {
	s = s.Scope("pbeam.MeanPerKey")
	// Obtain & validate type information from the underlying PCollection<K,V>.
	idT, kvT := beam.ValidateKVType(pcol.col)
	if kvT.Type() != reflect.TypeOf(kv.Pair{}) {
		log.Exitf("MeanPerKey must be used on a PrivatePCollection of type <K,V>, got type %v instead", kvT)
	}
	if pcol.codec == nil {
		log.Exitf("MeanPerKey: no codec found for the input PrivatePCollection.")
	}

	// Get privacy parameters.
	spec := pcol.privacySpec
	epsilon, delta, err := spec.consumeBudget(params.Epsilon, params.Delta)
	if err != nil {
		log.Exitf("couldn't consume budget: %v", err)
	}
	err = checkMeanPerKeyParams(params, epsilon, delta)
	if err != nil {
		log.Exit(err)
	}
	var noiseKind noise.Kind
	if params.NoiseKind == nil {
		noiseKind = noise.LaplaceNoise
		log.Infof("No NoiseKind specified, using Laplace Noise by default.")
	} else {
		noiseKind = params.NoiseKind.toNoiseKind()
	}

	// First, group together the privacy ID and the partition ID and do per-partition contribution bounding.
	// Result is PCollection<kv.Pair{ID,K},V>
	decoded := beam.ParDo(s,
		newPrepareMeanFn(idT, pcol.codec),
		pcol.col,
		beam.TypeDefinition{Var: beam.VType, T: pcol.codec.VType.T})

	maxContributionsPerPartition := getMaxContributionsPerPartition(params.MaxContributionsPerPartition)
	decoded = boundContributions(s, decoded, maxContributionsPerPartition)

	// Convert value to float64.
	// Result is PCollection<kv.Pair{ID,K},float64>.
	_, valueT := beam.ValidateKVType(decoded)
	convertFn, err := findConvertToFloat64Fn(valueT)
	if err != nil {
		log.Exit(err)
	}
	converted := beam.ParDo(s, convertFn, decoded)

	// Combine all values for <id, partition> into a slice.
	// Result is PCollection<kv.Pair{ID,K},[]float64>.
	combined := beam.CombinePerKey(s,
		&expandValuesCombineFn{},
		converted)

	// Result is PCollection<ID, pairArrayFloat64>.
	maxPartitionsContributed := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	rekeyed := beam.ParDo(s, rekeyArrayFloat64Fn, combined)
	// Do cross-partition contribution bounding.
	rekeyed = boundContributions(s, rekeyed, maxPartitionsContributed)

	// Now that the cross-partition contribution bounding is done, remove the privacy keys and decode the values.
	// Result is PCollection<partition, []float64>.
	partialPairs := beam.DropKey(s, rekeyed)
	partitionT := pcol.codec.KType.T
	partialKV := beam.ParDo(s,
		newDecodePairArrayFloat64Fn(partitionT),
		partialPairs,
		beam.TypeDefinition{Var: beam.XType, T: partitionT})

	// Compute the mean for each partition. Result is PCollection<partition, float64>.
	means := beam.CombinePerKey(s,
		newBoundedMeanFloat64Fn(epsilon, delta, maxPartitionsContributed, params.MaxContributionsPerPartition, params.MinValue, params.MaxValue, noiseKind),
		partialKV)
	// Finally, drop thresholded partitions.
	return beam.ParDo(s, dropThresholdedPartitionsFloat64Fn, means)
}

func checkMeanPerKeyParams(params MeanParams, epsilon, delta float64) error {
	err := checks.CheckEpsilon("pbeam.MeanPerKey", epsilon)
	if err != nil {
		return err
	}
	err = checks.CheckDeltaStrict("pbeam.MeanPerKey", delta)
	if err != nil {
		return err
	}
	err = checks.CheckBoundsFloat64("pbeam.MeanPerKey", params.MinValue, params.MaxValue)
	if err != nil {
		return err
	}
	return checks.CheckMaxPartitionsContributed("pbeam.MeanPerKey", params.MaxPartitionsContributed)
}

// decodePairArrayFloat64Fn transforms a PCollection<pairArrayFloat64<codedX,[]float64>> into a
// PCollection<X,[]float64>.
type decodePairArrayFloat64Fn struct {
	XType beam.EncodedType
	xDec  beam.ElementDecoder
}

func newDecodePairArrayFloat64Fn(t reflect.Type) *decodePairArrayFloat64Fn {
	return &decodePairArrayFloat64Fn{XType: beam.EncodedType{t}}
}

func (fn *decodePairArrayFloat64Fn) Setup() {
	fn.xDec = beam.NewElementDecoder(fn.XType.T)
}

func (fn *decodePairArrayFloat64Fn) ProcessElement(pair pairArrayFloat64) (beam.X, []float64) {
	x, err := fn.xDec.Decode(bytes.NewBuffer(pair.X))
	if err != nil {
		log.Exitf("pbeam.decodePairArrayFloat64Fn.ProcessElement: couldn't decode pair %v: %v", pair, err)
	}
	return x, pair.M
}

// findConvertFn gets the correct conversion to int64 or float64 function.
func findConvertToFloat64Fn(t typex.FullType) (interface{}, error) {
	switch t.Type().String() {
	case "int":
		return convertIntToFloat64Fn, nil
	case "int8":
		return convertInt8ToFloat64Fn, nil
	case "int16":
		return convertInt16ToFloat64Fn, nil
	case "int32":
		return convertInt32ToFloat64Fn, nil
	case "int64":
		return convertInt64ToFloat64Fn, nil
	case "uint":
		return convertUintToFloat64Fn, nil
	case "uint8":
		return convertUint8ToFloat64Fn, nil
	case "uint16":
		return convertUint16ToFloat64Fn, nil
	case "uint32":
		return convertUint32ToFloat64Fn, nil
	case "uint64":
		return convertUint64ToFloat64Fn, nil
	case "float32":
		return convertFloat32ToFloat64Fn, nil
	case "float64":
		return convertFloat64ToFloat64Fn, nil
	default:
		return nil, fmt.Errorf("pbeam.findConvertFn: unexpected value type %v", t)
	}
}

func convertIntToFloat64Fn(z beam.Z, i int) (beam.Z, float64) {
	return z, float64(i)
}
func convertInt8ToFloat64Fn(z beam.Z, i int8) (beam.Z, float64) {
	return z, float64(i)
}
func convertInt16ToFloat64Fn(z beam.Z, i int16) (beam.Z, float64) {
	return z, float64(i)
}
func convertInt32ToFloat64Fn(z beam.Z, i int32) (beam.Z, float64) {
	return z, float64(i)
}
func convertInt64ToFloat64Fn(z beam.Z, i int64) (beam.Z, float64) {
	return z, float64(i)
}
func convertUintToFloat64Fn(z beam.Z, i uint) (beam.Z, float64) {
	return z, float64(i)
}
func convertUint8ToFloat64Fn(z beam.Z, i uint8) (beam.Z, float64) {
	return z, float64(i)
}
func convertUint16ToFloat64Fn(z beam.Z, i uint16) (beam.Z, float64) {
	return z, float64(i)
}
func convertUint32ToFloat64Fn(z beam.Z, i uint32) (beam.Z, float64) {
	return z, float64(i)
}
func convertUint64ToFloat64Fn(z beam.Z, i uint64) (beam.Z, float64) {
	return z, float64(i)
}

type expandValuesAccum struct {
	Values []float64
}

type expandValuesCombineFn struct{}

func (fn *expandValuesCombineFn) CreateAccumulator() expandValuesAccum {
	return expandValuesAccum{Values: make([]float64, 0)}
}

func (fn *expandValuesCombineFn) AddInput(a expandValuesAccum, value float64) expandValuesAccum {
	a.Values = append(a.Values, value)
	return a
}

func (fn *expandValuesCombineFn) MergeAccumulators(a, b expandValuesAccum) expandValuesAccum {
	a.Values = append(a.Values, b.Values...)
	return a
}

func (fn *expandValuesCombineFn) ExtractOutput(a expandValuesAccum) []float64 {
	return a.Values
}

// prepareMeanFn takes a PCollection<ID,kv.Pair{K,V}> as input, and returns a
// PCollection<kv.Pair{ID,K},V>; where ID has been coded, and V has been
// decoded.
type prepareMeanFn struct {
	IDType         beam.EncodedType
	idEnc          beam.ElementEncoder
	InputPairCodec *kv.Codec
}

func newPrepareMeanFn(idType typex.FullType, kvCodec *kv.Codec) *prepareMeanFn {
	return &prepareMeanFn{
		IDType:         beam.EncodedType{idType.Type()},
		InputPairCodec: kvCodec,
	}
}

func (fn *prepareMeanFn) Setup() error {
	fn.idEnc = beam.NewElementEncoder(fn.IDType.T)
	return fn.InputPairCodec.Setup()
}

func (fn *prepareMeanFn) ProcessElement(id beam.W, pair kv.Pair) (kv.Pair, beam.V) {
	var idBuf bytes.Buffer
	if err := fn.idEnc.Encode(id, &idBuf); err != nil {
		log.Exitf("pbeam.prepareMeanFn.ProcessElement: couldn't encode ID %v: %v", id, err)
	}
	_, v := fn.InputPairCodec.Decode(pair)
	return kv.Pair{idBuf.Bytes(), pair.K}, v
}

// pairArrayFloat64 contains an encoded value and a slice of float64 metrics.
type pairArrayFloat64 struct {
	X []byte
	M []float64
}

// rekeyArrayFloat64Fn transforms a PCollection<kv.Pair<codedK,codedV>,[]float64> into a
// PCollection<codedK,pairArrayFloat64<codedV,[]float64>>.
func rekeyArrayFloat64Fn(kv kv.Pair, m []float64) ([]byte, pairArrayFloat64) {
	return kv.K, pairArrayFloat64{kv.V, m}
}

type boundedMeanAccumFloat64 struct {
	BM *dpagg.BoundedMeanFloat64
	SP *dpagg.PreAggSelectPartition
}

// boundedMeanFloat64Fn is a differentially private combineFn for obtaining mean of values. Do not
// initialize it yourself, use newBoundedMeanFloat64Fn to create a boundedMeanFloat64Fn instance.
type boundedMeanFloat64Fn struct {
	// Privacy spec parameters (set during initial construction).
	EpsilonNoise                 float64
	EpsilonPartitionSelection    float64
	DeltaNoise                   float64
	DeltaPartitionSelection      float64
	MaxPartitionsContributed     int64
	MaxContributionsPerPartition int64
	Lower                        float64
	Upper                        float64
	NoiseKind                    noise.Kind
	noise                        noise.Noise // Set during Setup phase according to NoiseKind.
}

// newBoundedMeanFloat6464Fn returns a boundedMeanFloat64Fn with the given budget and parameters.
func newBoundedMeanFloat64Fn(epsilon, delta float64, maxPartitionsContributed, maxContributionsPerPartition int64, lower, upper float64, noiseKind noise.Kind) *boundedMeanFloat64Fn {
	fn := &boundedMeanFloat64Fn{
		MaxPartitionsContributed:     maxPartitionsContributed,
		MaxContributionsPerPartition: maxContributionsPerPartition,
		Lower:                        lower,
		Upper:                        upper,
		NoiseKind:                    noiseKind,
	}
	fn.EpsilonNoise = epsilon / 2
	fn.EpsilonPartitionSelection = epsilon / 2
	switch noiseKind {
	case noise.GaussianNoise:
		fn.DeltaNoise = delta / 2
		fn.DeltaPartitionSelection = delta / 2
	case noise.LaplaceNoise:
		fn.DeltaNoise = 0
		fn.DeltaPartitionSelection = delta
	default:
		// TODO: return error instead
		log.Exitf("newBoundedMeanFloat64Fn: unknown noise.Kind (%v) is specified. Please specify a valid noise.", noiseKind)
	}
	return fn
}

func (fn *boundedMeanFloat64Fn) Setup() {
	fn.noise = noise.ToNoise(fn.NoiseKind)
}

func (fn *boundedMeanFloat64Fn) CreateAccumulator() boundedMeanAccumFloat64 {
	return boundedMeanAccumFloat64{
		BM: dpagg.NewBoundedMeanFloat64(&dpagg.BoundedMeanFloat64Options{
			Epsilon:                      fn.EpsilonNoise,
			Delta:                        fn.DeltaNoise,
			MaxPartitionsContributed:     fn.MaxPartitionsContributed,
			MaxContributionsPerPartition: fn.MaxContributionsPerPartition,
			Lower:                        fn.Lower,
			Upper:                        fn.Upper,
			Noise:                        fn.noise,
		}),
		SP: dpagg.NewPreAggSelectPartition(&dpagg.PreAggSelectPartitionOptions{
			Epsilon:                  fn.EpsilonPartitionSelection,
			Delta:                    fn.DeltaPartitionSelection,
			MaxPartitionsContributed: fn.MaxPartitionsContributed,
		}),
	}
}

func (fn *boundedMeanFloat64Fn) AddInput(a boundedMeanAccumFloat64, values []float64) boundedMeanAccumFloat64 {
	// We can have multiple values for each (privacy_key, partition_key) pair.
	// We need to add each value to BoundedMean as input but we need to add a single input
	// for each privacy_key to SelectPartition.
	for _, v := range values {
		a.BM.Add(v)
	}
	a.SP.Add()
	return a
}

func (fn *boundedMeanFloat64Fn) MergeAccumulators(a, b boundedMeanAccumFloat64) boundedMeanAccumFloat64 {
	a.BM.Merge(b.BM)
	a.SP.Merge(b.SP)
	return a
}

func (fn *boundedMeanFloat64Fn) ExtractOutput(a boundedMeanAccumFloat64) *float64 {
	if a.SP.Result() {
		result := a.BM.Result()
		return &result
	}
	return nil
}

func (fn *boundedMeanFloat64Fn) String() string {
	return fmt.Sprintf("%#v", fn)
}
