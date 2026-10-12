# Security policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's private vulnerability reporting
("Security" tab, "Report a vulnerability") on this repository.

## Scope notes

brepgraph parses CAD files with OpenCASCADE, a large native C++ library. Treat files from untrusted
sources like any other untrusted binary input: run conversions in a sandbox or a throwaway worker process.
Saved `.npz` graphs are loaded with `allow_pickle=False` and never execute code.
