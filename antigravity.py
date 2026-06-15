import random
from typing import List, Tuple


def build_anti_gravity_curve(start_point: Tuple[int, int], width: int, ceiling: int, severity: float) -> List[Tuple[int, int]]:
    """Builds a floating anomaly curve that simulates anti-gravity plastic extrusion.

    The generated filament line drifts upward and outward, creating a visually
    distinct anomaly mode for the SmartFlow simulator.
    """
    curve = [start_point]
    cx, cy = start_point
    segment_count = random.randint(6, 14)
    lateral_range = max(8, width // 2)
    upward_bias = min(25, 8 + int(severity * 22))

    for _ in range(segment_count):
        dx = random.randint(-lateral_range, lateral_range)
        dy = random.randint(-upward_bias, 8)
        cx += dx
        cy += dy
        cx = max(10, min(630, cx))
        cy = max(15, min(ceiling, cy))
        curve.append((cx, cy))

    return curve
