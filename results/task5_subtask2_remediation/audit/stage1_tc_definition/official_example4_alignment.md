# Official Example 4 alignment

Reference repository: `TriangleDataCenter/Triangle-BBH`  
Verified remote `main`: `13476c18cae4bac2623c6023829500194333b00e`  
Example 4 notebook blob: `58f18c5658331430729ed6ce1995e19e391f92fa`

## Window semantics

Example 4 selects the symmetric baseline with:

```python
abs(data_time / DAY - injected_parameters["coalescence_time"]) < 2.5
```

Accordingly, this project keeps the catalog `coalescence_time` fixed as the
window-selection anchor. The asymmetric task window changes only the offsets to
four days before and one day after that same catalog value.

## Inference semantics

Example 4 then changes parameterisation. It uses:

```python
WaveformGeneratorFRef
FDTDIResponseGeneratorFRef
FstatisticsFref
ParamDict2ParamArrFref
fref = 1e-3
tref_at_constellation = True
```

The inferred time and phase are `reference_time` and `reference_phase`. Before
passing injection metadata to Bilby/NESSAI, Example 4 removes catalog
`coalescence_time` and `coalescence_phase` and assigns `None` to both FRef truth
fields. Therefore the catalog time must not be drawn as an Example 4
`reference_time` truth.

## Current project consequence

The CPU reference branch preserves the official FRef semantics. The accelerated
GPU branch currently uses direct merger parameters and
`tc_at_constellation=True`; it is recorded as a distinct parameterisation. This
stage does not silently equate its `coalescence_time` with the catalog time and
does not choose a final SSB/constellation convention before the closed-loop
tests.

