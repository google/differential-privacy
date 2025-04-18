# Privacy on Beam

Privacy on Beam is an end-to-end differential privacy solution built on
[Apache Beam](https://beam.apache.org/documentation/).
It is intended to be usable by all developers, regardless of their differential
privacy expertise.

Internally, Privacy on Beam relies on the lower-level building blocks from the
differential privacy library and combines them into an "out-of-the-box" solution
that takes care of all the steps that are essential to differential privacy,
including noise addition, [partition selection](https://arxiv.org/abs/2006.03684),
and contribution bounding. Thus, rather than using the lower-level differential
privacy library, it is recommended to use Privacy on Beam, as it can reduce
implementation mistakes.

Privacy on Beam is only available in Go at the moment.

## How to Use

Our [codelab](https://codelabs.developers.google.com/codelabs/privacy-on-beam/)
about computing private statistics with Privacy on Beam
demonstrates how to use the library. Source code for the codelab is available in
the [codelab/](codelab)
directory.

Full documentation of the API is available as [godoc](https://godoc.org/github.com/google/differential-privacy/privacy-on-beam/v4/pbeam).

## Using with the "go" Command

For building Privacy on Beam with the ["go" command](https://golang.org/cmd/go/),
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

If you wish to run the codelab, you can do so by:
```shell
cd codelab/main
go run -mod=mod . -example=count -input_file=day_data.csv -output_stats_file=stats.csv -output_chart_file=chart.png
```

Change `example` to run other examples. See the
[codelab documentation](https://codelabs.developers.google.com/codelabs/privacy-on-beam/)
for more information.

Both for `go run` and `go test`, if you already built the code with `go build`,
you can omit `-mod=mod`.
