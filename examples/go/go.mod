module github.com/google/differential-privacy/examples/go

go 1.21

require (
	github.com/golang/glog v1.2.0
	github.com/google/differential-privacy/go/v3 v3.0.0-local // a nonexistent version number
)

require (
	golang.org/x/exp v0.0.0-20231226003508-02704c960a9b // indirect
	gonum.org/v1/gonum v0.14.0 // indirect
)

// To ensure the main branch works with the go tool when checked out locally.
replace github.com/google/differential-privacy/go/v3 v3.0.0-local => ../../go // see https://golang.org/doc/modules/managing-dependencies#local_directory