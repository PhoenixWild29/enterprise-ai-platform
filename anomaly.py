"""anomaly.py - Classical anomaly detection for sensor readings.

High-frequency telemetry is a statistics problem, not a language problem.
This is a deterministic robust z-score (median and MAD) per sensor: cheap,
explainable, and insensitive to the outliers it is looking for. The language
model never judges whether a reading is abnormal; it only cites the result.
"""

import statistics
from collections import defaultdict
from typing import Any

ROBUST_Z_THRESHOLD = 3.5  # Iglewicz and Hoaglin's recommended cut-off for the modified z-score
MIN_READINGS = 8
_MAD_SCALE = 0.6745


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def robust_z_anomalies(
    rows: list[dict[str, Any]],
    value_column: str = "VALUE",
    sensor_column: str = "SENSOR_ID",
    time_column: str = "READING_TS",
    threshold: float = ROBUST_Z_THRESHOLD,
) -> dict[str, Any]:
    """Flag readings whose modified z-score exceeds the threshold, per sensor.
    Sensors with too few readings or no spread are reported as not assessed
    rather than as normal."""
    by_sensor: dict[str, list[tuple[Any, float]]] = defaultdict(list)
    for row in rows:
        value = _number(row.get(value_column))
        if value is not None:
            by_sensor[str(row.get(sensor_column, "all"))].append((row.get(time_column), value))

    anomalies: list[dict[str, Any]] = []
    not_assessed: list[str] = []
    for sensor, readings in sorted(by_sensor.items()):
        values = [v for _, v in readings]
        if len(values) < MIN_READINGS:
            not_assessed.append(sensor)
            continue
        median = statistics.median(values)
        mad = statistics.median(abs(v - median) for v in values)
        if mad == 0:
            not_assessed.append(sensor)
            continue
        for timestamp, value in readings:
            z = _MAD_SCALE * (value - median) / mad
            if abs(z) > threshold:
                anomalies.append({"sensor": sensor, "time": timestamp, "value": value, "median": median, "robust_z": round(z, 2)})
    return {
        "method": "modified z-score (median and MAD) per sensor",
        "threshold": threshold,
        "readings_scanned": sum(len(r) for r in by_sensor.values()),
        "sensors_not_assessed": not_assessed,
        "anomalies": sorted(anomalies, key=lambda a: -abs(a["robust_z"])),
    }
