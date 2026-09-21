"""Time labels and fixed data-window definitions for task 5 subtask 2.

The catalog coalescence time is the immutable anchor for selecting data. Its
reference frame is deliberately left separate from the time parameter used by
the inference model until the stage-1 closed-loop tests establish that frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite
from typing import Callable, Mapping


WINDOW_ANCHOR_PARAMETER = "coalescence_time"


@dataclass(frozen=True)
class WindowDefinition:
    before_days: float
    after_days: float
    source: str


@dataclass(frozen=True)
class TimeConversionResult:
    input_day: float
    output_day: float
    delay_seconds: float
    iterations: int
    converged: bool


WINDOW_DEFINITIONS = {
    "official_baseline": WindowDefinition(
        before_days=2.5,
        after_days=2.5,
        source="Triangle-BBH Example 4: abs(data_time / DAY - catalog_tc) < 2.5",
    ),
    "task_five_day": WindowDefinition(
        before_days=4.0,
        after_days=1.0,
        source="UCAS 2026 task 5 subtask 2: four days before and one day after catalog_tc",
    ),
}


def get_catalog_tc_day(parameters: Mapping[str, object]) -> float:
    """Return the catalog time used only to anchor the two required windows."""
    if WINDOW_ANCHOR_PARAMETER not in parameters:
        raise KeyError(f"Missing catalog parameter: {WINDOW_ANCHOR_PARAMETER}")
    value = float(parameters[WINDOW_ANCHOR_PARAMETER])
    if not isfinite(value):
        raise ValueError("Catalog coalescence_time must be finite")
    return value


def get_window_bounds(catalog_tc_day: float, mode: str) -> tuple[float, float]:
    """Return immutable task window bounds around the catalog time."""
    if mode not in WINDOW_DEFINITIONS:
        raise ValueError(f"Unknown window mode: {mode}")
    definition = WINDOW_DEFINITIONS[mode]
    return (
        float(catalog_tc_day) - definition.before_days,
        float(catalog_tc_day) + definition.after_days,
    )


def validate_window_bounds(
    catalog_tc_day: float,
    mode: str,
    start_day: float,
    end_day: float,
    *,
    atol: float = 1e-12,
) -> None:
    """Fail if code or cached metadata shifts a task-defined data window."""
    expected_start, expected_end = get_window_bounds(catalog_tc_day, mode)
    if not isclose(start_day, expected_start, abs_tol=atol, rel_tol=0.0):
        raise RuntimeError(
            f"{mode} start changed: expected {expected_start}, got {start_day}"
        )
    if not isclose(end_day, expected_end, abs_tol=atol, rel_tol=0.0):
        raise RuntimeError(
            f"{mode} end changed: expected {expected_end}, got {end_day}"
        )


def constellation_to_ssb_day(
    tc_constellation_day: float,
    delay_seconds_at_constellation_time: Callable[[float], float],
) -> TimeConversionResult:
    """Convert a constellation-centre time to SSB using Triangle-BBH's sign."""
    delay_seconds = float(delay_seconds_at_constellation_time(tc_constellation_day))
    if not isfinite(delay_seconds):
        raise ValueError("Constellation delay must be finite")
    tc_ssb_day = float(tc_constellation_day) - delay_seconds / 86400.0
    return TimeConversionResult(
        input_day=float(tc_constellation_day),
        output_day=tc_ssb_day,
        delay_seconds=delay_seconds,
        iterations=1,
        converged=True,
    )


def ssb_to_constellation_day(
    tc_ssb_day: float,
    delay_seconds_at_constellation_time: Callable[[float], float],
    *,
    tolerance_seconds: float = 1e-7,
    max_iterations: int = 50,
) -> TimeConversionResult:
    """Solve t_const = t_ssb + delay(t_const) by fixed-point iteration."""
    if tolerance_seconds <= 0.0:
        raise ValueError("tolerance_seconds must be positive")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")

    tc_constellation_day = float(tc_ssb_day)
    delay_seconds = float("nan")
    for iteration in range(1, max_iterations + 1):
        delay_seconds = float(
            delay_seconds_at_constellation_time(tc_constellation_day)
        )
        if not isfinite(delay_seconds):
            raise ValueError("Constellation delay must be finite")
        updated_day = float(tc_ssb_day) + delay_seconds / 86400.0
        if abs(updated_day - tc_constellation_day) * 86400.0 <= tolerance_seconds:
            return TimeConversionResult(
                input_day=float(tc_ssb_day),
                output_day=updated_day,
                delay_seconds=delay_seconds,
                iterations=iteration,
                converged=True,
            )
        tc_constellation_day = updated_day

    return TimeConversionResult(
        input_day=float(tc_ssb_day),
        output_day=tc_constellation_day,
        delay_seconds=delay_seconds,
        iterations=max_iterations,
        converged=False,
    )
