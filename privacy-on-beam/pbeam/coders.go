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
	"encoding/gob"
	"reflect"

	"github.com/apache/beam/sdks/go/pkg/beam"
)

// Coders for serializing DP Aggregation Accumulators.

func init() {
	beam.RegisterCoder(reflect.TypeOf(countAccum{}), encodeCountAccum, decodeCountAccum)
	beam.RegisterCoder(reflect.TypeOf(boundedSumAccumInt64{}), encodeBoundedSumAccumInt64, decodeBoundedSumAccumInt64)
	beam.RegisterCoder(reflect.TypeOf(boundedSumAccumFloat64{}), encodeBoundedSumAccumFloat64, decodeBoundedSumAccumFloat64)
	beam.RegisterCoder(reflect.TypeOf(boundedMeanAccumFloat64{}), encodeBoundedMeanAccumFloat64, decodeBoundedMeanAccumFloat64)
	beam.RegisterCoder(reflect.TypeOf(expandValuesAccum{}), encodeExpandValuesAccum, decodeExpandValuesAccum)
}

func encodeCountAccum(ca countAccum) ([]byte, error) {
	return encode(ca)
}

func decodeCountAccum(data []byte) (countAccum, error) {
	var ret countAccum
	err := decode(&ret, data)
	return ret, err
}

func encodeBoundedSumAccumInt64(v boundedSumAccumInt64) ([]byte, error) {
	return encode(v)
}

func decodeBoundedSumAccumInt64(data []byte) (boundedSumAccumInt64, error) {
	var ret boundedSumAccumInt64
	err := decode(&ret, data)
	return ret, err
}

func encodeBoundedSumAccumFloat64(v boundedSumAccumFloat64) ([]byte, error) {
	return encode(v)
}

func decodeBoundedSumAccumFloat64(data []byte) (boundedSumAccumFloat64, error) {
	var ret boundedSumAccumFloat64
	err := decode(&ret, data)
	return ret, err
}

func encodeBoundedMeanAccumFloat64(v boundedMeanAccumFloat64) ([]byte, error) {
	return encode(v)
}

func decodeBoundedMeanAccumFloat64(data []byte) (boundedMeanAccumFloat64, error) {
	var ret boundedMeanAccumFloat64
	err := decode(&ret, data)
	return ret, err
}

func encodeExpandValuesAccum(v expandValuesAccum) ([]byte, error) {
	return encode(v)
}

func decodeExpandValuesAccum(data []byte) (expandValuesAccum, error) {
	var ret expandValuesAccum
	err := decode(&ret, data)
	return ret, err
}

func encode(v interface{}) ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)
	err := enc.Encode(v)
	return buf.Bytes(), err
}

func decode(v interface{}, data []byte) error {
	return gob.NewDecoder(bytes.NewReader(data)).Decode(v)
}
