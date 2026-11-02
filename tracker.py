"""Small IoU-based tracker for stable IDs across nearby camera frames."""
from __future__ import annotations

from dataclasses import dataclass
import time

from detector import Detection


@dataclass
class _Track:
    track_id: int
    class_id: int
    box: tuple[float, float, float, float]
    last_seen: float


class IoUTracker:
    """Greedy same-class matching; intentionally dependency-free."""

    def __init__(self, iou_threshold: float = 0.25, max_age_seconds: float = 1.5):
        self.iou_threshold = iou_threshold
        self.max_age_seconds = max_age_seconds
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection], now: float | None = None) -> list[Detection]:
        now = time.monotonic() if now is None else now
        self._tracks = [t for t in self._tracks if now - t.last_seen <= self.max_age_seconds]
        available = set(range(len(self._tracks)))
        result: list[Detection] = []
        for detection in sorted(detections, key=lambda d: d.score, reverse=True):
            best_index, best_iou = None, self.iou_threshold
            current_box = (detection.x1, detection.y1, detection.x2, detection.y2)
            for index in available:
                track = self._tracks[index]
                if track.class_id != detection.class_id:
                    continue
                overlap = self._iou(current_box, track.box)
                if overlap > best_iou:
                    best_iou, best_index = overlap, index
            if best_index is None:
                track_id = self._next_id
                self._next_id += 1
                self._tracks.append(_Track(track_id, detection.class_id, current_box, now))
            else:
                track = self._tracks[best_index]
                available.remove(best_index)
                track_id = track.track_id
                # Mild smoothing reduces box jitter while following moving objects.
                previous = track.box
                smooth = tuple(0.35 * old + 0.65 * new for old, new in zip(previous, current_box))
                track.box = smooth
                track.last_seen = now
                current_box = smooth
            result.append(Detection(*current_box, detection.class_id, detection.score, track_id))
        return result

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        left, top = max(a[0], b[0]), max(a[1], b[1])
        right, bottom = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union > 0 else 0.0
