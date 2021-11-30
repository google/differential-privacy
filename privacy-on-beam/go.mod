module github.com/google/differential-privacy/privacy-on-beam

go 1.16

require (
	github.com/apache/beam v2.31.0+incompatible
	github.com/golang/glog v0.0.0-20160126235308-23def4e6c14b
	github.com/google/differential-privacy/go v0.0.0-local // a nonexistent version number
	github.com/google/go-cmp v0.5.5
	github.com/google/uuid v1.3.0 // indirect
	gonum.org/v1/plot v0.10.0
	google.golang.org/grpc v1.40.0 // indirect
	google.golang.org/protobuf v1.26.0
)

// To ensure the main branch works with the go tool when checked out locally.
// It would still not be possible to depend on the main branch with the go tool
// for now, i.e. building with the go tool after
// "go get github.com/google/differential-privacy/privacy-on-beam@main" will
// fail.
replace github.com/google/differential-privacy/go v0.0.0-local => ../go // see https://golang.org/doc/modules/managing-dependencies#local_directory