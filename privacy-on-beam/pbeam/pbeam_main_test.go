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

package pbeam

import (
	"testing"

	"github.com/apache/beam/sdks/v2/go/pkg/beam/testing/ptest"
)

func TestMain(m *testing.M) {
	ptest.MainWithDefault(m, "direct")
}

// Below are used in various tests.
var gaussianNoise = GaussianNoise{}

// Helper function to create a PrivacySpec that deals with error handling.
func privacySpec(t *testing.T, params PrivacySpecParams) *PrivacySpec {
	t.Helper()
	spec, err := NewPrivacySpec(params)
	if err != nil {
		t.Fatalf("Failed to create PrivacySpec")
	}
	return spec
}
