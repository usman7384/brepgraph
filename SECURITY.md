# Security policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private vulnerability reporting
("Security" tab, "Report a vulnerability") on this repository.

## Scope notes

brepgraph parses CAD files with OpenCASCADE, a large native C++ library. Treat files from untrusted
sources like any other untrusted binary input: run conversions in a sandbox or a throwaway worker process.
`load_many` and `brepgraph batch` already process files in worker processes, so a crash on a corrupt file is
reported for that file only; this isolates crashes, it does not sandbox the parser.
Saved `.npz` graphs are loaded with `allow_pickle=False` and never execute code. They do not contain the absolute
path of the source file unless it is requested with `save(..., include_source_path=True)`.
