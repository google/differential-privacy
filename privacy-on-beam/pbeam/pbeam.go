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

// Package pbeam provides an API for building differentially private data
// processing pipelines using Apache Beam (https://beam.apache.org) with its
// Go SDK (https://godoc.org/github.com/apache/beam/sdks/v2/go/pkg/beam).
//
// It introduces the concept of a PrivatePCollection, an interface mirroring
// Apache Beam's PCollection concept. PrivatePCollection implements additional
// restrictions and aggregations to facilitate differentially private analysis.
// This API is meant to be used by developers without differential privacy
// expertise.
//
// For a step-by-step introduction to differential privacy, Apache Beam, and
// example usage of this library, see:
// https://codelabs.developers.google.com/codelabs/privacy-on-beam/index.html;
// a codelab meant for developers who want to get started on using this library
// and generating differentially private metrics.
//
// The rest of this package-level comment goes into more detail about the
// precise guarantees offered by this API, and assumes some familiarity with
// the Apache Beam model, its Go SDK, and differential privacy.
//
// To understand the main API contract provided by PrivatePCollection, consider
// the following example pipeline.
//
//		p := beam.NewPipeline()
//		s := p.Root()
//		// The input is a series of files in which each line contains the data of a privacy unit (e.g. an individual).
//		input := textio.Read(s, "/path/to/files/*.txt") // input is a PCollection<string>
//		// Extracts the privacy ID and the data associated with each line: extractID is a func(string) (userID,data).
//		icol := beam.ParDo(s, input, extractID) // icol is a PCollection<privacyUnitID,data>
//		// Transforms the input PCollection into a PrivatePCollection with parameters ε=1 and δ=10⁻¹⁰.
//		// The privacy ID is "hidden" by the operation: pcol behaves as if it were a PCollection<data>.
//	  spec, err := pbeam.NewPrivacySpec(pbeam.PrivacySpecParams{
//	    AggregationEpsilon: 0.5,
//	    PartitionSelectionEpsilon: 0.5,
//	    PartitionSelectionDelta: 1e-10,
//	  })
//		pcol := pbeam.MakePrivate(s, icol, spec) // pcol is a PrivatePCollection<data>
//		// Arbitrary transformations can be applied to the data…
//		pcol = pbeam.ParDo(s, pcol, someDoFn)
//		pcol = pbeam.ParDo(s, pcol, otherDoFn)
//		// …and to retrieve PCollection outputs, differentially private aggregations must be used.
//		// For example, assuming pcol is now a PrivatePCollection<field,float64>:
//		sumParams := pbeam.SumParams{MaxPartitionsContributed: 10, MaxValue: 5}
//		ocol := pbeam.SumPerKey(s, pcol2, sumParams) // ocol is a PCollection<field,float64>
//		// And it is now possible to output this data.
//		textio.Write(s, "/path/to/output/file", ocol)
//
// The behavior of PrivatePCollection is similar to the behavior of PCollection.
// In particular, it implements arbitrary per-record transformations via ParDo.
// However, the contents of a PrivatePCollection cannot be written to disk.
// For example, there is no equivalent of:
//
//	textio.Write(s, "/path/to/output/file", pcol)
//
// In order to retrieve data encapsulated in a PrivatePCollection, it is
// necessary to use one of the differentially private aggregations provided with
// this library (e.g., count, sum, mean), which transforms the
// PrivatePCollection back into a PCollection.
//
// This is because of the API contract provided by this library: once data is
// encapsulated in a PrivatePCollection, all its outputs are differentially
// private. More precisely, suppose a PrivatePCollection pcol is created from a
// PCollection<K,V> icol with privacy parameters (ε,δ), and output in one or
// several PCollections (ocol1, ocol2, ocol3). Let f be the corresponding
// randomized transformation, associating icol with (ocol1, ocol2, ocol3). Then
// f is (ε,δ)-differentially private in the following sense. Let icol' be the
// PCollection obtained by removing all records associated with a given value of
// K in icol. Then, for any set S of possible outputs:
//
//	P[f(icol) ∈ S] ≤ exp(ε) * P[f(icol') ∈ S] + δ.
//
// The K, in the example above, is userID, representing a user identifier. This
// means that the full list of contributions of any given user is protected. However, this does not need
// to be the case; the protected property might be different than a user
// identifier. In this library, we use the more general terminology of "privacy
// unit" to refer to the type of this identifier (for example, user ID, event
// ID, a pair (user ID, day)); and "privacy identifier" to refer to a
// particular instance of this identifier (for example, user n°4217, event n°99,
// or the pair (user n°4127,2020-06-24)).
//
// Note that the interface contract of PrivatePCollection has limitations. this
// library assumes that the user of the library is trusted with access to the
// underlying raw data. This intended user is a well-meaning developer trying to
// produce anonymized metrics about data using differential privacy. The API
// tries to make it easy to anonymize metrics that are safe to publish to
// untrusted parties; and difficult to break the differential privacy privacy
// guarantees by mistake.
//
// However, this API does not attempt to protect against malicious library
// users. In particular, nothing prevents a user of this library from adding a
// side-effect to a ParDo function to leak raw data and bypass differential
// privacy guarantees. Similarly, ParDo functions are allowed to return errors
// that crash the pipeline, which could be abused to leak raw data. There is no
// protection against timing or side-channel attacks, as we assume that the only
// thing malicious users have access to is the output data.
package pbeam

import (
	"bytes"
	"fmt"
	"math"
	"reflect"
	"strings"
	"sync"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/v3/checks"
	"github.com/google/differential-privacy/go/v3/noise"
	"github.com/google/differential-privacy/privacy-on-beam/v3/internal/kv"
	"github.com/apache/beam/sdks/v2/go/pkg/beam"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/core/typex"
	"github.com/apache/beam/sdks/v2/go/pkg/beam/register"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func init() {
	register.DoFn2x3[beam.U, kv.Pair, beam.U, beam.V, error](&dropKeyFn{})
	register.DoFn2x3[beam.U, kv.Pair, beam.U, beam.W, error](&dropValueFn{})
	register.DoFn1x3[beam.V, string, beam.V, error](&extractStructFieldFn{})
	register.DoFn1x3[beam.V, string, beam.V, error](&extractProtoFieldFn{})
}

// PrivacySpec contains information about the privacy parameters used in
// a PrivatePCollection. It encapsulates a privacy budget that must be shared
// between all aggregations on PrivatePCollections using this PrivacySpec. If
// you have multiple pipelines in the same binary, and want them to use
// different privacy budgets, call NewPrivacySpec multiple times and give a
// different PrivacySpec to each PrivatePCollection.
type PrivacySpec struct {
	aggregationBudget        *privacyBudget // Epsilon/Delta (ε,δ) budget available for aggregations performed on this PrivatePCollection.
	partitionSelectionBudget *privacyBudget // Epsilon/Delta (ε,δ) budget available for partition selections performed on this PrivatePCollection.
	preThreshold             int64          // Pre-threshold K applied on top of DP partition selection.
	testMode TestMode // Used for test pipelines, disabled by default.
}

// PartitionSelectionParams holds the ε & δ budget to be used for private partition selection of
// an aggregation. It is also used to specify parameters of a SelectPartitions aggregation.
type PartitionSelectionParams struct {
	// Differential privacy budget consumed by private partition selection.
	//
	// If this is the only private partition selection operation in the pipeline (e.g. the only
	// aggregation in the pipeline, the only aggregation in the pipeline where public partitions are
	// not specified, the only SelectPartitions aggregation aggregation in the pipeline where other
	// aggregations use public partitions), both Epsilon and Delta can be left 0; in that case, the
	// entire budget reserved for partition selection in the PrivacySpec is consumed.
	Epsilon, Delta float64
	// Warning: This parameter can currently only be set for SelectPartitions aggregation.
	//
	// The maximum number of distinct keys that a given privacy identifier can influence. If a privacy
	// identifier is associated to more keys, random keys will be dropped. There is an inherent
	// trade-off when choosing this parameter: a larger MaxPartitionsContributed leads to less data
	// loss due to contribution bounding, but since the noise added in aggregations is scaled
	// according to maxPartitionsContributed, it also means that probability of keeping a partition
	// with a given privacy ID count is lowered.
	//
	// Required.
	MaxPartitionsContributed int64
}

// PrivacySpecParams contains parameters to construct a PrivacySpec.
//
// Uses the new privacy budget API where clients specify aggregation budget and partition selection budget separately.
type PrivacySpecParams struct {
	// Epsilon (ε) budget available for aggregations performed on this PrivatePCollection. Required unless
	// the only aggregation in the pipeline is pbeam.SelectPartitions.
	AggregationEpsilon float64
	// Delta (δ) budget available for aggregations performed on this PrivatePCollection. Only set it if you
	// use Gaussian Noise.
	AggregationDelta float64
	// Epsilon (ε) budget available for partition selections performed on this PrivatePCollection. Required unless
	// you use public partitions.
	PartitionSelectionEpsilon float64
	// Delta (δ) budget available for partition selections performed on this PrivatePCollection. Required unless
	// you use public partitions.
	PartitionSelectionDelta float64
	// PreThreshold contains an optional additional threshold. Pre-thresholding is
	// performed in combination with private partition selection to ensure that
	// each partition has at least a K number of unique contributions.
	//
	// See https://github.com/google/differential-privacy/blob/main/common_docs/pre_thresholding.md
	// for more information.
	//
	// Pre-thresholding is currently only available for partition selection.
	PreThreshold int64
	// Test mode for test pipelines, disabled by default. Set it to TestModeWithContributionBounding or
	// TestModeWithoutContributionBounding if you want to enable test mode.
	TestMode TestMode
}
type privacyBudget struct {
	// Epsilon/Delta (ε,δ) budget available.
	epsilon, delta    float64
	partiallyConsumed bool       // Whether some budget has already been consumed from this privacy budget.
	mux               sync.Mutex // To avoid race conditions on epsilon & delta.
}

// consumes a differential privacy budget (ε,δ) from a PrivacySpec. If epsilon and delta are 0,
// it consumes the entire budget, which is only possible if this is the first time its budget is consumed.
//
// Returns the budget consumed.
func (budget *privacyBudget) consume(epsilon, delta float64) (eps, del float64, err error) {
	budget.mux.Lock()
	defer budget.mux.Unlock()
	eps, del, err = budget.getThreadUnsafe(epsilon, delta)
	budget.epsilon = budget.epsilon - eps
	budget.delta = budget.delta - del
	budget.partiallyConsumed = true
	return eps, del, err
}

// get computes the differential privacy budget (ε,δ) to consume from a PrivacySpec.If epsilon and
// delta are 0, it gets the entire available budget, which is only possible if this is the first
// time its budget is to be consumed.
//
// Returns the budget to consume.
//
// Warning: use consumeBudget to actually consume the budget.
func (budget *privacyBudget) get(epsilon, delta float64) (eps, del float64, err error) {
	budget.mux.Lock()
	defer budget.mux.Unlock()
	return budget.getThreadUnsafe(epsilon, delta)
}

// getThreadUnsafe is not thread-safe and should not be used directly. Instead, use get or consume.
func (budget *privacyBudget) getThreadUnsafe(epsilon, delta float64) (eps, del float64, err error) {
	if epsilon == 0 && delta == 0 {
		return budget.getEntireBudget()
	}
	return budget.getPartialBudget(epsilon, delta)
}

func (budget *privacyBudget) getEntireBudget() (eps, del float64, err error) {
	if budget.partiallyConsumed {
		return 0, 0, fmt.Errorf("trying to consume entire budget of PrivacySpec, but it has already been partially or fully consumed: %+v ", budget)
	}
	return budget.epsilon, budget.delta, nil
}

func (budget *privacyBudget) getPartialBudget(epsilon, delta float64) (eps, del float64, err error) {
	if budgetSlightlyTooLarge(budget.epsilon, epsilon) {
		log.Infof("corrected rounding error for epsilon budget allocation (requested: %f, available: %f, difference: %e)", epsilon, budget.epsilon, epsilon-budget.epsilon)
		epsilon = budget.epsilon
	}
	if budgetSlightlyTooLarge(budget.delta, delta) {
		log.Infof("corrected rounding error for delta budget allocation (requested: %e, available: %e, difference: %e)", delta, budget.delta, delta-budget.delta)
		delta = budget.delta
	}
	if budget.epsilon < epsilon || budget.delta < delta {
		return 0, 0, fmt.Errorf("not enough budget left for PrivacySpec: trying to consume epsilon=%f and delta=%e out of remaining epsilon=%f and delta=%e. Did you forget to split your budget among aggregations?", epsilon, delta, budget.epsilon, budget.delta)
	}
	return epsilon, delta, nil
}

// Relative tolerance of the budget that is assumed to be a rounding error and
// will consume all remaining budget.
const eqBudgetRelTol = 1e9

// budgetSlightlyTooLarge returns true if and only if requested is slightly larger
// than remaining, i.e. requested is larger by remaining up to a rounding error
// (computed as remaining/eqBudgetRelTol).
func budgetSlightlyTooLarge(remaining, requested float64) bool {
	diff := remaining - requested
	if diff >= 0 {
		return false
	}
	return math.Abs(diff) <= remaining/eqBudgetRelTol
}

// NoiseKind represents the kind of noise to be used in an aggregations.
type NoiseKind interface {
	toNoiseKind() noise.Kind
}

// GaussianNoise is an aggregations param that makes them use Gaussian Noise.
type GaussianNoise struct{}

func (gn GaussianNoise) toNoiseKind() noise.Kind {
	return noise.GaussianNoise
}

// LaplaceNoise is an aggregations param that makes them use Laplace Noise.
type LaplaceNoise struct{}

func (ln LaplaceNoise) toNoiseKind() noise.Kind {
	return noise.LaplaceNoise
}

// NewPrivacySpec creates a new PrivacySpec with the specified privacy budget
// and parameters.
//
// Aggregation(Epsilon|Delta) and PartitionSelection(Epsilon|Delta) are the total
// (ε,δ)-differential privacy budget for the pipeline. If there is only one aggregation
// or partition selection,  the entire budget will be used for this operation. Otherwise,
// the user must specify how the privacy budget is split across aggregations.
func NewPrivacySpec(params PrivacySpecParams) (*PrivacySpec, error) {

	err := checks.CheckEpsilon(params.AggregationEpsilon)
	if err != nil {
		return nil, fmt.Errorf("AggregationEpsilon: %v", err)
	}
	err = checks.CheckDelta(params.AggregationDelta)
	if err != nil {
		return nil, fmt.Errorf("AggregationDelta: %v", err)
	}
	err = checks.CheckEpsilon(params.PartitionSelectionEpsilon)
	if err != nil {
		return nil, fmt.Errorf("PartitionSelectionEpsilon: %v", err)
	}
	err = checks.CheckDelta(params.PartitionSelectionDelta)
	if err != nil {
		return nil, fmt.Errorf("PartitionSelectionDelta: %v", err)
	}
	if params.PreThreshold > 0 && params.PartitionSelectionDelta == 0 {
		return nil, fmt.Errorf("when PreThreshold is set, partition selection budget must also be set")
	}
	err = checks.CheckPreThreshold(params.PreThreshold)
	if err != nil {
		return nil, fmt.Errorf("PreThreshold: %v", err)
	}
	if params.AggregationEpsilon == 0 && params.PartitionSelectionEpsilon == 0 {
		return nil, fmt.Errorf("either AggregationEpsilon or PartitionSelectionEpsilon must be set to a positive value")
	}
	if params.PartitionSelectionEpsilon != 0 && params.PartitionSelectionDelta == 0 {
		return nil, fmt.Errorf("PartitionSelectionDelta must be set to a positive value whenever PartitionSelectionEpsilon is set. "+
			"PartitionSelectionEpsilon is currently set to (%f)", params.PartitionSelectionEpsilon)
	}
	return &PrivacySpec{
		aggregationBudget:        &privacyBudget{epsilon: params.AggregationEpsilon, delta: params.AggregationDelta},
		partitionSelectionBudget: &privacyBudget{epsilon: params.PartitionSelectionEpsilon, delta: params.PartitionSelectionDelta},
		preThreshold:             params.PreThreshold,
		testMode:                 params.TestMode,
	}, nil
}

// A PrivatePCollection embeds a PCollection, associating each element to a
// privacy identifier, and ensures that its content can only be written to a
// sink after being anonymized using differentially private aggregations.
//
// We call "privacy identifier" the value of the identifier associated with a
// record (e.g. 62934947), and "privacy unit" the semantic type of this
// identifier (e.g. "user ID"). Typical choices for privacy units include user
// IDs or session IDs. This choice determines the privacy unit protected by
// differential privacy. For example, if the privacy unit is user ID, then the
// output of aggregations will be (ε,δ)-indistinguishable from the output
// obtained via PrivatePCollection in which all records associated with a
// single user ID have been removed, or modified.
//
// Some operations on PCollections are also available on PrivatePCollection,
// for example a limited subset of ParDo operations. They transparently
// propagate privacy identifiers, preserving the privacy guarantees of the
// PrivatePCollection.
type PrivatePCollection struct {
	// PCollection<ID,X>, where ID is the privacy unit
	col beam.PCollection
	// If this PrivatePCollection is of <K,V> type, we store each pair as a
	// kv.Pair; and this is the codec that can be used to decode it.
	codec *kv.Codec
	// Privacy budget and parameters attached to this PrivatePCollection
	privacySpec *PrivacySpec
}

// MakePrivate transforms a PCollection<K,V> into a PrivatePCollection<V>,
// where <K> is the privacy unit.
func MakePrivate(_ beam.Scope, col beam.PCollection, spec *PrivacySpec) PrivatePCollection {
	if !typex.IsKV(col.Type()) {
		log.Fatalf("MakePrivate: PCollection col=%v  must be of KV type", col)
	}
	return PrivatePCollection{
		col:         col,
		privacySpec: spec,
	}
}

// MakePrivateFromStruct creates a PrivatePCollection from a PCollection of
// structs and the qualified path (seperated by ".") of the struct field to
// use as a privacy key.
// For example:
//
//	  type exampleStruct1 struct {
//	    IntField int
//			 StructField exampleStruct2
//	  }
//
//	  type  exampleStruct2 struct {
//	    StringField string
//	  }
//
// If col is a PCollection of exampleStruct1, you could use "IntField" or
// "StructField.StringField" as idFieldPath.
//
// # Caution
//
// The privacy key field must be a simple type (e.g. int, string, etc.), or
// a pointer to a simple type and all its parents must be structs or
// pointers to structs.
//
// If the privacy key field is not set, all elements without a set field
// will be attributed to the same (default) privacy unit, likely degrading utility
// of future DP aggregations. Similarly, if the idFieldPath or any of its
// parents are nil, those elements will be attributed to the same (default)
// privacy unit as well.
func MakePrivateFromStruct(s beam.Scope, col beam.PCollection, spec *PrivacySpec, idFieldPath string) PrivatePCollection {
	s = s.Scope("pbeam.MakePrivateFromStruct")
	msgTypex := col.Type()
	if typex.IsKV(msgTypex) {
		log.Fatalf("MakePrivateFromStruct: PCollection col=%v cannot be of KV type", col)
	}
	msgType := msgTypex.Type()
	if msgType.Kind() != reflect.Struct {
		log.Fatalf("MakePrivateFromStruct: PCollection col=%v must be composed of structs", col)
	}
	extractFn := &extractStructFieldFn{IDFieldPath: idFieldPath}
	return PrivatePCollection{
		col:         beam.ParDo(s, extractFn, col),
		privacySpec: spec,
	}
}

type extractStructFieldFn struct {
	IDFieldPath string
}

func (ext *extractStructFieldFn) ProcessElement(v beam.V) (string, beam.V, error) {
	idField, err := ext.getIDField(v)
	if err != nil {
		return "", nil, fmt.Errorf("couldn't retrieve ID field %s: %v", ext.IDFieldPath, err)
	}
	// We use %#v to guarantee two different keys map to different strings
	return fmt.Sprintf("%#v", idField), v, nil
}

// getIDField retrieves the ID field (specified by the IDFieldPath) from
// struct or pointer to a struct s.
func (ext *extractStructFieldFn) getIDField(s any) (any, error) {
	subFieldNames := strings.Split(ext.IDFieldPath, ".")
	subField := reflect.ValueOf(s)
	var subFieldPath bytes.Buffer
	for _, subFieldName := range subFieldNames {
		subField = ext.getPointedValue(subField) // Retrieve the pointed value if subField is a pointer, no-op otherwise.
		if subField.Kind() != reflect.Struct {
			return nil, fmt.Errorf("%s (%v) should be a struct or a pointer to a struct", subFieldPath.String(), subField.Kind())
		}
		subField = subField.FieldByName(subFieldName)
		subFieldPath.WriteString(subFieldName + ".")
		if !subField.IsValid() {
			return nil, fmt.Errorf("no such field %s (%v) in s", subFieldPath.String(), subField.Kind())
		}
	}
	subField = ext.getPointedValue(subField) // Retrieve the  pointed value if subField is a pointer, no-op otherwise.
	if err := ext.checkSimpleType(subField); err != nil {
		return nil, err
	}
	// TODO Set the ID field to default value.
	return subField.Interface(), nil
}

// getPointedValue returns the value pointed by v if v is a pointer. If v is nil,
// it returns the default value for the type pointed by v. If v is not a pointer,
// it returns v.
func (ext *extractStructFieldFn) getPointedValue(v reflect.Value) reflect.Value {
	zeroVal := reflect.Value{}
	if reflect.Indirect(v) != zeroVal {
		return reflect.Indirect(v)
	}
	return reflect.Zero(v.Type().Elem())
}

func (ext *extractStructFieldFn) checkSimpleType(v reflect.Value) error {
	switch v.Kind() {
	case reflect.Bool, reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint,
		reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Float32, reflect.Float64,
		reflect.Complex64, reflect.Complex128, reflect.String:
		return nil
	default:
		return fmt.Errorf("ID field must be a simple type (e.g. int, string), got type %v instead", v.Kind())
	}
}

// MakePrivateFromProto creates a PrivatePCollection from a PCollection of
// proto messages and the qualified name of the field to use as a privacy key.
// The field and all its parents must be non-repeated, and the field itself
// cannot be a submessage.
func MakePrivateFromProto(s beam.Scope, col beam.PCollection, spec *PrivacySpec, idFieldPath string) PrivatePCollection {
	s = s.Scope("pbeam.MakePrivateFromProto")
	msgTypex := col.Type()
	if typex.IsKV(msgTypex) {
		log.Fatalf("MakePrivateFromProto: PCollection col=%v  cannot be of KV type", col)
	}
	msgType := msgTypex.Type()
	var sampleMessage proto.Message
	if !msgType.Implements(reflect.TypeOf(&sampleMessage).Elem()) {
		log.Fatalf("MakePrivateFromProto: PCollection col=%v  must be composed of proto messages", col)
	}
	extractFn := &extractProtoFieldFn{
		IDFieldPath: idFieldPath,
		MsgType:     beam.EncodedType{msgType},
	}
	return PrivatePCollection{
		col:         beam.ParDo(s, extractFn, col),
		privacySpec: spec,
	}
}

type extractProtoFieldFn struct {
	IDFieldPath string
	MsgType     beam.EncodedType
	desc        protoreflect.MessageDescriptor
}

func (ext *extractProtoFieldFn) ProcessElement(v beam.V) (string, beam.V, error) {
	pb := v.(proto.Message)
	reflectPb := pb.ProtoReflect()
	// If ext.desc hasn't been initialized, initialize it now.
	if ext.desc == nil {
		ext.desc = reflectPb.Descriptor()
	}
	idField, err := ext.extractField(reflectPb)
	if err != nil {
		return "", nil, fmt.Errorf("couldn't extract field %s from proto: %w", ext.IDFieldPath, err)
	}
	out := reflectPb.Interface()
	return fmt.Sprint(idField), out, nil
}

// extractProtoField retrieves the value of a protoreflect.Message field based on
// its fully qualified name, and deletes this field from the original message.
// It fails if the field is a submessage, if it is repeated, or if any of its
// parents are repeated.
func (ext *extractProtoFieldFn) extractField(pb protoreflect.Message) (any, error) {
	parts := strings.Split(ext.IDFieldPath, ".")
	curPb := pb
	curDesc := ext.desc
	for i, part := range parts {
		fieldDesc := curDesc.Fields().ByName((protoreflect.Name)(part))
		if fieldDesc == nil {
			return nil, fmt.Errorf("couldn't get field %s from the proto message", strings.Join(parts[:i+1], "."))
		}
		switch {
		case fieldDesc.Cardinality() == protoreflect.Repeated:
			return nil, fmt.Errorf("repeated field %s found in the proto message", strings.Join(parts[:i+1], "."))
		case fieldDesc.Kind() == protoreflect.MessageKind || fieldDesc.Kind() == protoreflect.GroupKind:
			// Continue looking into subfields.
			curDesc = fieldDesc.Message()
			if curPb.Has(fieldDesc) {
				curPb = curPb.Get(fieldDesc).Message()
			} else {
				curPb = curPb.NewField(fieldDesc).Message()
			}
		default:
			// Remove and return the (value) field as a string.
			value := curPb.Get(fieldDesc).String()
			// TODO Remove the ID field.
			return value, nil
		}
	}
	return nil, fmt.Errorf("submessage field %s found in the proto message", ext.IDFieldPath)
}

// DropKey drops the key for an input PrivatePCollection<K,V>. It returns
// a PrivatePCollection<V>.
func DropKey(s beam.Scope, pcol PrivatePCollection) PrivatePCollection {
	pcol.col = beam.ParDo(s, &dropKeyFn{pcol.codec}, pcol.col, beam.TypeDefinition{Var: beam.VType, T: pcol.codec.VType.T})
	pcol.codec = nil
	return pcol
}

type dropKeyFn struct {
	Codec *kv.Codec
}

func (fn *dropKeyFn) Setup() {
	fn.Codec.Setup()
}

func (fn *dropKeyFn) ProcessElement(id beam.U, kv kv.Pair) (beam.U, beam.V, error) {
	_, v, err := fn.Codec.Decode(kv)
	return id, v, err
}

// DropValue drops the value for an input PrivatePCollection<K,V>. It returns
// a PrivatePCollection<K>.
func DropValue(s beam.Scope, pcol PrivatePCollection) PrivatePCollection {
	pcol.col = beam.ParDo(s, &dropValueFn{pcol.codec}, pcol.col, beam.TypeDefinition{Var: beam.WType, T: pcol.codec.KType.T})
	pcol.codec = nil
	return pcol
}

type dropValueFn struct {
	Codec *kv.Codec
}

func (fn *dropValueFn) Setup() {
	fn.Codec.Setup()
}

func (fn *dropValueFn) ProcessElement(id beam.U, kv kv.Pair) (beam.U, beam.W, error) {
	k, _, err := fn.Codec.Decode(kv)
	return id, k, err
}
