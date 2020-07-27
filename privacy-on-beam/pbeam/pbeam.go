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
// Go SDK (https://godoc.org/github.com/apache/beam/sdks/go/pkg/beam).
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
//  p := beam.NewPipeline()
//  s := p.Root()
//  // The input is a series of files in which each line contains the data of a user.
//  input := textio.Read(s, "/path/to/files/*.txt") // input is a PCollection<string>
//  // Extracts the user ID and the data associated with each line: extractID is a func(string) (userID,data).
//  icol := beam.ParDo(s, input, extractID) // icol is a PCollection<userID,data>
//  // Transforms the input PCollection into a PrivatePCollection with parameters ε=1 and δ=10⁻¹⁰.
//  // The user ID is "hidden" by the operation: pcol behaves as if it were a PCollection<data>.
//  pcol := MakePrivate(s, icol, NewPrivacySpec(1, 1e-10)) // pcol is a PrivatePCollection<data>
//  // Arbitrary transformations can be applied to the data…
//  pcol = ParDo(s, pcol, someDoFn)
//  pcol = ParDo(s, pcol, otherDoFn)
//  // …and to retrieve PCollection outputs, differentially private aggregations must be used.
//  // For example, assuming pcol is now a PrivatePCollection<field,float64>:
//  sumParams := SumParams{MaxPartitionsContributed: 10, MaxValue: 5}
//  ocol := SumPerKey(s, pcol2, sumParams) // ocol is a PCollection<field,float64>
//  // And it is now possible to output this data.
//  textio.Write(s, "/path/to/output/file", ocol)
//
// The behavior of PrivatePCollection is similar to the behavior of PCollection.
// In particular, it implements arbitrary per-record transformations via ParDo.
// However, the contents of a PrivatePCollection cannot be written to disk.
// For example, there is no equivalent of:
//
//  textio.Write(s, "/path/to/output/file", pcol)
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
//  P[f(icol) ∈ S] ≤ exp(ε) * P[f(icol') ∈ S] + δ.
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
	"reflect"
	"strings"
	"sync"

	log "github.com/golang/glog"
	"github.com/google/differential-privacy/go/noise"
	"github.com/google/differential-privacy/privacy-on-beam/internal/kv"
	"github.com/apache/beam/sdks/go/pkg/beam"
	"github.com/apache/beam/sdks/go/pkg/beam/core/typex"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/reflect/protoreflect"
)

func init() {
	beam.RegisterType(reflect.TypeOf((*extractProtoFieldFn)(nil)))
	beam.RegisterType(reflect.TypeOf((*extractStructFieldFn)(nil)))
	// TODO: add tests to make sure we don't forget anything here
}

// PrivacySpec contains information about the privacy parameters used in
// a PrivatePCollection. It encapsulates a privacy budget that must be shared
// between all aggregations on PrivatePCollections using this PrivacySpec. If
// you have multiple pipelines in the same binary, and want them to use
// different privacy budgets, call NewPrivacySpec multiple times and give a
// different PrivacySpec to each PrivatePCollection.
type PrivacySpec struct {
	epsilon           float64 // ε budget available for this PrivatePCollection.
	delta             float64 // δ budget available for this PrivatePCollection.
	partiallyConsumed bool    // Whether some privacy budget has already been consumed from this PrivacySpec.
	mux sync.Mutex
}

// consumeBudget consumes a differential privacy budget (ε,δ) from a
// PrivacySpec. If epsilon and delta are 0, it consumes the entire budget,
// which is only possible if this is the first time its budget is consumed.
// Returns the budget consumed.
func (ps *PrivacySpec) consumeBudget(epsilon, delta float64) (eps, del float64, err error) {
	ps.mux.Lock()
	defer ps.mux.Unlock()
	if epsilon == 0 && delta == 0 {
		return ps.consumeEntireBudget()
	}
	return ps.consumePartialBudget(epsilon, delta)
}

func (ps *PrivacySpec) consumeEntireBudget() (eps, del float64, err error) {
	if ps.partiallyConsumed {
		return 0, 0, fmt.Errorf("trying to consume entire budget of PrivacySpec, but it has already been partially or fully consumed: %+v ", ps)
	}
	eps, del = ps.epsilon, ps.delta
	ps.epsilon = 0
	ps.delta = 0
	ps.partiallyConsumed = true
	return eps, del, nil
}

func (ps *PrivacySpec) consumePartialBudget(epsilon, delta float64) (eps, del float64, err error) {
	if budgetSlightlyTooLarge(ps.epsilon, epsilon) {
		log.Infof("corrected rounding error for epsilon budget allocation (requested: %f, available: %f, difference: %e)", epsilon, ps.epsilon, epsilon-ps.epsilon)
		epsilon = ps.epsilon
	}
	if budgetSlightlyTooLarge(ps.delta, delta) {
		log.Infof("corrected rounding error for delta budget allocation (requested: %e, available: %e, difference: %e)", epsilon, ps.epsilon, epsilon-ps.epsilon)
		delta = ps.delta
	}
	if ps.epsilon < epsilon || ps.delta < delta {
		return 0, 0, fmt.Errorf("not enough budget left for PrivacySpec: trying to consume epsilon=%f and delta=%e out of %+v", epsilon, delta, ps)
	}
	ps.epsilon -= epsilon
	ps.delta -= delta
	ps.partiallyConsumed = true
	return epsilon, delta, nil
}

// Relative tolerance of the budget that is assumed to be a rounding error and
// will consume all remaining budget.
const eqBudgetRelTol = 1e9

func budgetSlightlyTooLarge(remaining, requested float64) bool {
	if requested < remaining {
		return false
	}
	return remaining-requested <= remaining/eqBudgetRelTol
}

// PrivacySpecOption is used for customizing PrivacySpecs. In the typical use
// case, PrivacySpecOptions are passed into the NewPrivacySpec constructor to
// create a further customized PrivacySpec.
type PrivacySpecOption interface {
	updatePrivacySpec(ps *PrivacySpec)
}

// getMaxPartitionsContributed returns a maxPartitionsContributed parameter
// if it greater than zero, otherwise it fails.
func getMaxPartitionsContributed(spec *PrivacySpec, maxPartitionsContributed int64) int64 {
	if maxPartitionsContributed <= 0 {
		// TODO: return error instead
		log.Exitf("MaxPartitionsContributed must be set to a positive value.")
	}
	return maxPartitionsContributed
}

// getMaxContributionsPerPartition returns a maxContributionsPerPartition parameter
// if it greater than zero, otherwise it fails.
func getMaxContributionsPerPartition(maxContributionsPerPartition int64) int64 {
	if maxContributionsPerPartition <= 0 {
		// TODO: return error instead
		log.Exitf("MaxContributionsPerPartition must be set to a positive value.")
	}
	return maxContributionsPerPartition
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
// and options.
//
// The epsilon and delta arguments are the total (ε,δ)-differential privacy
// budget for the pipeline. If there is only one aggregation, the entire budget
// will be used for this aggregation. Otherwise, the user must specify how the
// privacy budget is split across aggregations.
func NewPrivacySpec(epsilon, delta float64, options ...PrivacySpecOption) *PrivacySpec {
	ps := &PrivacySpec{
		epsilon: epsilon,
		delta:   delta,
	}
	for _, opt := range options {
		opt.updatePrivacySpec(ps)
	}
	return ps
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
		log.Exitf("MakePrivate: PCollection must be of KV type: %v", col)
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
//   type exampleStruct1 struct {
//     IntField int
//		 StructField exampleStruct2
//   }
//
//   type  exampleStruct2 struct {
//     StringField string
//   }
//
// If col is a PCollection of exampleStruct1, you could use "IntField" or
// "StructField.StringField" as idFieldPath.
//
// Caution
//
// The privacy key field must be a simple type (e.g. int, string, etc.), or
// a pointer to a simple type and all its parents must be structs or
// pointers to structs.
//
// If the privacy key field is not set, all elements without a set field
// will be attributed to the same (default) user, likely degrading utility
// of future DP aggregations. Similarly, if the idFieldPath or any of its
// parents are nil, those elements will be attributed to the same (default)
// user as well.
func MakePrivateFromStruct(s beam.Scope, col beam.PCollection, spec *PrivacySpec, idFieldPath string) PrivatePCollection {
	s = s.Scope("pbeam.MakePrivateFromStruct")
	msgTypex := col.Type()
	if typex.IsKV(msgTypex) {
		log.Exitf("MakePrivateFromStruct: PCollection cannot be of KV type: %v", col)
	}
	msgType := msgTypex.Type()
	if msgType.Kind() != reflect.Struct {
		log.Exitf("MakePrivateFromStruct: PCollection must be composed of structs", col)
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
		return "", nil, fmt.Errorf("Couldn't retrieve ID field %s: %v", ext.IDFieldPath, err)
	}
	// We use %#v to guarantee two different keys map to different strings
	return fmt.Sprintf("%#v", idField), v, nil
}

// getIDField retrieves the ID field (specified by the IDFieldPath) from
// struct or pointer to a struct s.
func (ext *extractStructFieldFn) getIDField(s interface{}) (interface{}, error) {
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
		return fmt.Errorf("id field must be a simple type (e.g. int, string), got type %v instead", v.Kind())
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
		log.Exitf("MakePrivateFromProto: PCollection cannot be of KV type: %v", col)
	}
	msgType := msgTypex.Type()
	var dummyMessage proto.Message
	if !msgType.Implements(reflect.TypeOf(&dummyMessage).Elem()) {
		log.Exitf("MakePrivateFromProto: PCollection must be composed of proto messages", col)
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

func (ext *extractProtoFieldFn) ProcessElement(v beam.V) (string, beam.V) {
	pb := v.(proto.Message)
	reflectPb := pb.ProtoReflect()
	// If ext.desc hasn't been initialized, initialize it now.
	if ext.desc == nil {
		ext.desc = reflectPb.Descriptor()
	}
	idField, err := ext.extractField(reflectPb)
	if err != nil {
		log.Exitf("couldn't extract field %s from proto: %v", ext.IDFieldPath, err)
	}
	out := reflectPb.Interface()
	return fmt.Sprint(idField), out
}

// extractProtoField retrieves the value of a protoreflect.Message field based on
// its fully qualified name, and deletes this field from the original message.
// It fails if the field is a submessage, if it is repeated, or if any of its
// parents are repeated.
func (ext *extractProtoFieldFn) extractField(pb protoreflect.Message) (interface{}, error) {
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
