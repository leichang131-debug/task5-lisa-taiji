"""Initialise the non-computational part of the stage-1 time audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.time_conventions import (
    WINDOW_ANCHOR_PARAMETER,
    WINDOW_DEFINITIONS,
    get_catalog_tc_day,
    get_window_bounds,
    validate_window_bounds,
)


DEFAULT_PARAMETER_FILE = REPO_ROOT / "data" / "tdc" / "0_2_MBHB_parameters.h5"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "results"
    / "task5_subtask2_remediation"
    / "audit"
    / "stage1_tc_definition"
)


def git_head(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def read_catalog_parameter_file(path: Path) -> tuple[dict[str, float], dict[str, object]]:
    with h5py.File(path, "r") as handle:
        parameters = {
            name: float(handle[name][0])
            for name in handle.keys()
            if getattr(handle[name], "shape", None) == (1,)
        }
        attributes = {name: str(value) for name, value in handle.attrs.items()}
    return parameters, attributes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parameter-file", type=Path, default=DEFAULT_PARAMETER_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    parameters, hdf5_attributes = read_catalog_parameter_file(args.parameter_file)
    catalog_tc_day = get_catalog_tc_day(parameters)

    window_definitions = {}
    for mode, definition in WINDOW_DEFINITIONS.items():
        start_day, end_day = get_window_bounds(catalog_tc_day, mode)
        validate_window_bounds(catalog_tc_day, mode, start_day, end_day)
        window_definitions[mode] = {
            "before_days": definition.before_days,
            "after_days": definition.after_days,
            "start_day": start_day,
            "end_day": end_day,
            "duration_days": end_day - start_day,
            "source": definition.source,
        }

    output_dir = args.output_dir.resolve()
    for child in (
        "manifests",
        "conversions",
        "waveform_checks",
        "likelihood_checks",
        "figures",
    ):
        (output_dir / child).mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": 1,
        "stage": "stage1_tc_definition",
        "status": "initialised_no_closed_loop_tests_run",
        "repository_base_commit": git_head(REPO_ROOT),
        "parameter_file": str(args.parameter_file.resolve()),
        "parameter_file_root_attributes": hdf5_attributes,
        "window_anchor": {
            "parameter": WINDOW_ANCHOR_PARAMETER,
            "catalog_tc_day": catalog_tc_day,
            "purpose": "immutable data-selection anchor only",
            "reference_frame": "unverified_no_HDF5_metadata",
        },
        "window_definitions": window_definitions,
        "official_example4": {
            "parameterisation": "FRef",
            "time_parameter": "reference_time",
            "phase_parameter": "reference_phase",
            "fref_hz": 0.001,
            "tref_at_constellation": True,
            "catalog_coalescence_time_used_as_reference_time_truth": False,
        },
        "accelerated_gpu_route": {
            "parameterisation": "direct_merger",
            "time_parameter": "coalescence_time",
            "tc_at_constellation_current_value": True,
            "final_time_frame_decision": "pending_closed_loop_tests",
        },
    }
    (output_dir / "manifests" / "time_convention_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
