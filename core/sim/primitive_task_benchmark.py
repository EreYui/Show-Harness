"""Task-level benchmark helpers for atomic versus continuous Cartesian actions.

The benchmark deliberately uses state-feedback oracles.  Perception and planning are
removed so a success-rate gap measures the reachable task set of the action interface,
not VLM quality.  Isaac Lab runners feed the returned decision delta through the same
relative-IK environment and physics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PrimitiveTaskResult:
    task: str
    controller: str
    trial: int
    success: bool
    decisions: int
    final_error_m: float
    path_length_m: float
    constraint_pass_rate: float = 1.0
    max_constraint_error_m: float = 0.0
    endpoint_reached: bool = False
    target_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)


def atomic_delta(error, step_m: float) -> tuple[np.ndarray, str]:
    """Fixed-length, single-axis correction matching the six MVTOKEN moves."""
    delta = np.asarray(error, dtype=float)
    if delta.shape != (3,):
        raise ValueError("error must be XYZ")
    axis = int(np.argmax(np.abs(delta)))
    sign = 1.0 if delta[axis] >= 0.0 else -1.0
    out = np.zeros(3, dtype=float)
    out[axis] = sign * float(step_m)
    positive = ("MV_FWD", "MV_LEFT", "MV_UP")
    negative = ("MV_BACK", "MV_RIGHT", "MV_DOWN")
    return out, positive[axis] if sign > 0.0 else negative[axis]


def continuous_delta(error, step_m: float) -> tuple[np.ndarray, str]:
    """Arbitrary-direction Cartesian correction with a bounded step norm."""
    delta = np.asarray(error, dtype=float)
    if delta.shape != (3,):
        raise ValueError("error must be XYZ")
    distance = float(np.linalg.norm(delta))
    if distance <= 0.0:
        return np.zeros(3, dtype=float), "DIRECT_HOLD"
    scale = min(1.0, float(step_m) / distance)
    return delta * scale, "DIRECT_XYZ"


def path_length(points: Iterable) -> float:
    samples = np.asarray(list(points), dtype=float)
    if len(samples) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(samples, axis=0), axis=1).sum())


def line_constraint_errors(points: Iterable, start, end) -> np.ndarray:
    """Per-sample distance to the finite 3-D line segment from start to end."""
    samples = np.asarray(list(points), dtype=float)
    p0 = np.asarray(start, dtype=float)
    p1 = np.asarray(end, dtype=float)
    direction = p1 - p0
    denom = float(direction @ direction)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("points must be an Nx3 sequence")
    if denom <= 0.0:
        return np.linalg.norm(samples - p0, axis=1)
    amount = np.clip(((samples - p0) @ direction) / denom, 0.0, 1.0)
    nearest = p0 + amount[:, None] * direction
    return np.linalg.norm(samples - nearest, axis=1)


def summarize_task(
    *,
    task: str,
    controller: str,
    trial: int,
    trace,
    target,
    target_offset,
    decisions: int,
    endpoint_tolerance_m: float,
    constraint_errors=None,
    constraint_tolerance_m: float | None = None,
    required_constraint_pass_rate: float = 1.0,
) -> PrimitiveTaskResult:
    samples = np.asarray(trace, dtype=float)
    if len(samples) == 0:
        raise ValueError("trace must contain at least one position")
    final_error = float(np.linalg.norm(np.asarray(target, dtype=float) - samples[-1]))
    endpoint = final_error <= float(endpoint_tolerance_m)
    errors = (
        np.zeros(len(samples), dtype=float)
        if constraint_errors is None
        else np.asarray(constraint_errors, dtype=float)
    )
    if len(errors) != len(samples):
        raise ValueError("constraint_errors length must match trace")
    if constraint_tolerance_m is None:
        pass_rate = 1.0
    else:
        pass_rate = float(np.mean(errors <= float(constraint_tolerance_m)))
    success = bool(endpoint and pass_rate >= float(required_constraint_pass_rate))
    return PrimitiveTaskResult(
        task=str(task),
        controller=str(controller),
        trial=int(trial),
        success=success,
        decisions=int(decisions),
        final_error_m=final_error,
        path_length_m=path_length(samples),
        constraint_pass_rate=pass_rate,
        max_constraint_error_m=float(errors.max()) if len(errors) else 0.0,
        endpoint_reached=endpoint,
        target_offset_m=tuple(float(x) for x in target_offset),
    )


__all__ = [
    "PrimitiveTaskResult",
    "atomic_delta",
    "continuous_delta",
    "line_constraint_errors",
    "path_length",
    "summarize_task",
]
