# Local verification

Verified on Windows with Python 3.12.4. Direct package versions are recorded in `requirements-tested.txt`.

## Completed checks

- All five command-line entry points return successfully for `--help`.
- All Python files parse successfully.
- Thirteen numerical unit tests pass using in-memory inputs.
- The archived numerical function bodies are unchanged after source selection and usage-text cleanup. AST comparison excludes the argument-parser function whose input/output example help text was made portable.
- End-to-end execution succeeds for denoising, two-epoch deformation, section splitting, longitudinal summary, and matched monitoring accuracy.
- The runtime check uses a 5,184-point synthetic cylindrical surface and a target with 0.1 m radial convergence. It verifies more than 80% valid estimates and a signed median within 0.015 m of the imposed convergence. These are software checks, not manuscript accuracy results.
- Runtime test inputs and generated outputs are stored outside the release directory and excluded from the ZIP.
- The release source contains no retained user-specific drive paths. Pattern screening found no common GitHub access tokens or private-key blocks.

## Scope

The test run used the current installed environment, not a newly downloaded clean environment. The dependency ranges and other platforms have not all been tested. No original field point-cloud run was performed, and the checks do not establish reproduction of the manuscript's numerical results.

## Re-run

```bash
python -m unittest discover -s tests -v
```

Tests cover sign convention, unchanged epochs, unavailable neighborhoods, median/percentile weighting, joint noise constraints, exclusion of self from radius counts, valid-only aggregation, interval boundaries, empty intervals, zero reference values, and matched reference/location sets.
