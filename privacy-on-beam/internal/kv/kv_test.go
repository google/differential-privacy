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

package kv

import (
	"reflect"
	"testing"

	"github.com/apache/beam/sdks/v2/go/pkg/beam"
)

func TestNewCodec(t *testing.T) {
	for _, tc := range []struct {
		desc  string
		kType reflect.Type
		vType reflect.Type
		want  *Codec
	}{
		{"calling NewCodec with (KType, VType) (int, int)", reflect.TypeOf(int(0)), reflect.TypeOf(int(0)), &Codec{KType: beam.EncodedType{T: reflect.TypeOf(int(0))}, VType: beam.EncodedType{T: reflect.TypeOf(int(0))}}},
		{"calling NewCodec with (KType, VType) (int, nil)", reflect.TypeOf(int(0)), nil, nil},
		{"calling NewCodec with (KType, VType) (nil, int)", nil, reflect.TypeOf(int(0)), nil},
		{"calling NewCodec with (KType, VType) (nil, nil)", nil, nil, nil},
	} {

		if c := NewCodec(tc.kType, tc.vType); !reflect.DeepEqual(c, tc.want) {
			t.Errorf("When %s, expected %v, got %v instead.", tc.desc, tc.want, c)
		}
	}
}

func TestCodec_Setup(t *testing.T) {
	for _, tc := range []struct {
		desc    string
		c       *Codec
		wantErr bool
	}{
		{"properly initialized codec", NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf(int(0))), false},
		{"not initialized codec", &Codec{}, true},
		{"codec missing VType", &Codec{KType: beam.EncodedType{T: reflect.TypeOf(int(0))}}, true},
		{"codec missing KType", &Codec{VType: beam.EncodedType{T: reflect.TypeOf(int(0))}}, true},
	} {
		if err := tc.c.Setup(); (err != nil) != tc.wantErr {
			t.Errorf("When %s, wantErr=%t, got=%v", tc.desc, tc.wantErr, err)
		}
	}
}

func TestCodec_Encode_Decode(t *testing.T) {
	c := NewCodec(reflect.TypeOf(int(0)), reflect.TypeOf("str"))
	c.Setup()
	inputK, inputV := 3, "x"
	pair, err := c.Encode(inputK, inputV)
	if err != nil {
		t.Fatal(err)
	}
	outputK, outputV, err := c.Decode(pair)
	if err != nil {
		t.Fatal(err)
	}
	if inputK != outputK || inputV != outputV {
		t.Errorf("Expected (%v, %v) but got (%v, %v) instead.", inputK, inputV, outputK, outputV)
	}
}
