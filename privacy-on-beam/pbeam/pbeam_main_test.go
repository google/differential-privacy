package pbeam

import (
	"testing"

	"github.com/apache/beam/sdks/go/pkg/beam/testing/ptest"
)

func TestMain(m *testing.M) {
	ptest.Main(m)
}

// Used in various tests.
var gaussianNoise = GaussianNoise{}
