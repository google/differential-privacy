module github.com/google/differential-privacy/examples/go

go 1.25

require (
	github.com/golang/glog v1.2.5
	github.com/google/differential-privacy/go/v4 v4.0.0
)

// Use the current version of the Go DP Library.
replace github.com/google/differential-privacy/go/v4 => ../../go

require gonum.org/v1/gonum v0.16.0 // indirect
