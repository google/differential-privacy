module github.com/google/differential-privacy/examples/go

go 1.22

require (
	github.com/golang/glog v1.2.0
	github.com/google/differential-privacy/go/v4 v4.0.0
)


// Use the current version of the Go DP Library.
replace github.com/google/differential-privacy/go/v4 => ../../go

require (
	golang.org/x/exp v0.0.0-20240222234643-814bf88cf225 // indirect
	gonum.org/v1/gonum v0.14.0 // indirect
)
