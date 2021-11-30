module github.com/google/differential-privacy/examples/go

go 1.16

require (
	github.com/golang/glog v1.0.0
	github.com/google/differential-privacy/go v0.0.0-local // a nonexistent version number
)

// To ensure the main branch works with the go tool when checked out locally.
replace github.com/google/differential-privacy/go v0.0.0-local => ../../go // see https://golang.org/doc/modules/managing-dependencies#local_directory