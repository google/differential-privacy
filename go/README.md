# Differential Privacy in Go

This is the Go implementation of the differential privacy library. For general
details and key definitions, see the top-level documentation.
This document describes Go-specific aspects.

## How to Use

Usage of the Go Differential Privacy library is demonstrated in the
[codelab](../examples/go/).

Full documentation of the API is available as [godoc](https://godoc.org/github.com/google/differential-privacy/go/v2/dpagg).

## Using with the "go" Command

For building the differential privacy library with the ["go" command](https://golang.org/cmd/go/),
you can run the following:
```shell
go build -mod=mod ./...
```
This will build all the packages. `-mod=mod` is necessary for installing all the
dependencies automatically. Otherwise, you'll be asked to install each
dependency manually.

Similarly, you can run all the tests with:
```shell
go test -mod=mod ./...
```

If you already built the code with `go build`, you can omit `-mod=mod`.

If you wish to run the
[codelab](../examples/go/),
you can do so by (from the root of the repository):

```shell
cd examples/go/main
go run -mod=mod . -scenario=CountVisitsPerHour -input_file=../data/day_data.csv -non_private_output_file=out1.csv -private_output_file=out2.csv
```

Change `scenario` and `input_file` to run other scenarios. Check out the
[README](../examples/go/README.md)
for more information.
