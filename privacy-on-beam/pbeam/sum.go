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
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/go/pkg/beam/transforms/stats"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*prepareSumFn)(nil)))
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

// SumPerKey sums the values associated with each key in a
// PrivatePCollection<K,V>, adding differentially private noise to the sums and
// doing pre-aggregation thresholding to remove sums with a low number of
// distinct privacy identifiers.
//
// Note: Do not use when your results may cause overflows for Int64 and Float64
// values. This aggregation is not hardened for such applications yet.
//
// SumPerKey transforms a PrivatePCollection<K,V> either into a
// PCollection<K,int64> or a PCollection<K,float64>, depending on whether its
// input is an integer type or a float type.
func SumPerKey(s beam.Scope, pcol PrivatePCollection, params SumParams) beam.PCollection {
	s = s.Scope("pbeam.SumPerKey")
	// Obtain & validate type information from the underlying PCollection<K,V>.
	idT, kvT := beam.ValidateKVType(pcol.col)
	if kvT.Type() != reflect.TypeOf(kv.Pair{}) {
		log.Exitf("SumPerKey must be used on a PrivatePCollection of type <K,V>, got type %v instead", kvT)
	}
	if pcol.codec == nil {
		log.Exitf("SumPerKey: no codec found for the input PrivatePCollection.")
	}

	// Get privacy parameters.
	spec := pcol.privacySpec
	epsilon, delta, err := spec.consumeBudget(params.Epsilon, params.Delta)
	if err != nil {
		log.Exitf("couldn't consume budget: %v", err)
	}
	err = checkSumPerKeyParams(params, epsilon, delta)
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
	maxPartitionsContributed := getMaxPartitionsContributed(spec, params.MaxPartitionsContributed)
	// First, group together the privacy ID and the partition ID, and sum the
	// values per-user and per-partition.
	decoded := beam.ParDo(s,
		newPrepareSumFn(idT, pcol.codec),
		pcol.col,
		beam.TypeDefinition{Var: beam.VType, T: pcol.codec.VType.T})
	summed := stats.SumPerKey(s, decoded)
	// Second, convert the sum to int64 or float64, and re-key.
	_, sumT := beam.ValidateKVType(summed)
	convertFn, err := findConvertFn(sumT)
	if err != nil {
		log.Exit(err)
	}
	vKind, err := getKind(convertFn)
	if err != nil {
		log.Exit(err)
	}
	converted := beam.ParDo(s, convertFn, summed)
	rekeyed := beam.ParDo(s, findRekeyFn(vKind), converted)
	// Third, do per-user contribution bounding.
	rekeyed = boundContributions(s, rekeyed, maxPartitionsContributed)
	// Fourth, now that contribution bounding is done, remove the privacy keys,
	// decode the value, and do a DP sum with all the partial sums.
	partialSumPairs := beam.DropKey(s, rekeyed)
	partitionT := pcol.codec.KType.T
	partialSumKV := beam.ParDo(s,
		newDecodePairFn(partitionT, vKind),
		partialSumPairs,
		beam.TypeDefinition{Var: beam.XType, T: partitionT})
	sums := beam.CombinePerKey(s,
		newBoundedSumFn(epsilon, delta, maxPartitionsContributed, params.MinValue, params.MaxValue, noiseKind, vKind),
		partialSumKV)
	// Drop thresholded partitions.
	sums = beam.ParDo(s, findDropThresholdedPartitionsFn(vKind), sums)
	// Clamp negative counts to zero when MinValue is non-negative.
	if params.MinValue >= 0 {
		sums = beam.ParDo(s, findClampNegativePartitionsFn(vKind), sums)
	}
	return sums
}

func checkSumPerKeyParams(params SumParams, epsilon, delta float64) error {
	err := checks.CheckEpsilon("pbeam.SumPerKey", epsilon)
	if err != nil {
		return err
	}
	err = checks.CheckDeltaStrict("pbeam.SumPerKey", delta)
	if err != nil {
		return err
	}
	err = checks.CheckBoundsFloat64("pbeam.SumPerKey", params.MinValue, params.MaxValue)
	if err != nil {
		return err
	}
	return checks.CheckMaxPartitionsContributed("pbeam.SumPerKey", params.MaxPartitionsContributed)
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

func (fn *prepareSumFn) ProcessElement(id beam.W, pair kv.Pair) (kv.Pair, beam.V) {
	var idBuf bytes.Buffer
	if err := fn.idEnc.Encode(id, &idBuf); err != nil {
		log.Exitf("pbeam.prepareSumFn.ProcessElement: couldn't encode ID %v: %v", id, err)
	}
	_, v := fn.InputPairCodec.Decode(pair)
	return kv.Pair{idBuf.Bytes(), pair.K}, v
}

// findConvertFn gets the correct conversion to int64 or float64 function.
func findConvertFn(t typex.FullType) (interface{}, error) {

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
		return nil, fmt.Errorf("pbeam.findConvertFn: unexpected value type %v", t)
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
func getKind(fn interface{}) (reflect.Kind, error) {
	if fn == nil {
		return reflect.Invalid, fmt.Errorf("pbeam.getKind: fn is nil, should be a convertFn")
	}
	if reflect.TypeOf(fn).Kind() != reflect.Func {
		return reflect.Invalid, fmt.Errorf("pbeam.getKind: fn is %v, should be a function", reflect.TypeOf(fn).Kind())
	}
	if reflect.TypeOf(fn).NumOut() < 2 {
		return reflect.Invalid, fmt.Errorf("pbeam.getKind: fn has %v outputs, expected at least 2", reflect.TypeOf(fn).NumOut())
	}
	return reflect.TypeOf(fn).Out(1).Kind(), nil
}
