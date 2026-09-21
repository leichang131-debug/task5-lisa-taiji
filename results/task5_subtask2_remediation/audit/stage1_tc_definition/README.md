# Stage 1: time-definition audit

This directory contains the stage-1 audit for the coalescence/reference-time
definitions. Initialisation is intentionally non-computational: it does not run
F-statistics, Fisher analysis, heterodyned likelihoods, or NESSAI.

The catalog `coalescence_time` is frozen as the data-window anchor:

- official Example 4 baseline: `catalog_tc - 2.5` to `catalog_tc + 2.5` days;
- UCAS task window: `catalog_tc - 4` to `catalog_tc + 1` days.

This does not assert that the catalog value is an SSB or constellation-centre
inference parameter. The HDF5 file has no reference-frame metadata, so that
source-data interpretation remains explicitly unresolved.

## Closed-loop result

The stage-1 tests passed on an NVIDIA GeForce RTX 4060 Laptop GPU. The time
round trip covered the direct position, reflected position, and three seeded
random sky positions. The waveform check covered the official symmetric
5-day window and the task-specific 4+1-day window, each with interpolation
disabled and enabled.

- maximum SSB/constellation round-trip error: `0.0 s` (limit `1e-3 s`);
- maximum relative waveform L2 residual: `0.0` (limit `1e-6`);
- maximum absolute noise-weighted mismatch: `0.0` (limit `1e-6`);
- full-likelihood difference for all four waveform cases: `0.0`.

These results verify the software contract: passing SSB `tc` with
`tc_at_constellation=False` is numerically identical to first converting it
to constellation time and passing it with `tc_at_constellation=True`. They do
not establish the undocumented reference frame of the catalog field.

Subdirectories:

- `manifests/`: machine-readable conventions and official-reference audit;
- `conversions/`: reserved for SSB/constellation round-trip checks;
- `waveform_checks/`: reserved for direct-response equivalence checks;
- `likelihood_checks/`: reserved for minimal full-likelihood checks;
- `figures/`: reserved for stage-1 diagnostic figures.

Regenerate the manifest from the real TDC parameter file with:

```bash
python scripts/stage1_initialize_time_audit.py
```

Run the time round-trip and GPU waveform-equivalence checks with all caches on
the E drive:

```bash
export TASK5_STAGE1_CACHE_DIR=/mnt/e/TDCEnv/Cache/task5-stage1
python scripts/stage1_tc_closed_loop.py
```

The script treats the catalog value as a numerical SSB seed only for the
equivalence identity; it does not assert that the TDC catalog metadata is SSB.
It tests five sky positions for the time conversion and both required data
windows with interpolation disabled and enabled for the GPU response.
