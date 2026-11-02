"""Lightweight YOLOv8 ONNX inference using OpenCV DNN (no PyTorch at runtime)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from labels_ar import COCO_NAMES_AR


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    score: float
    track_id: int = 0

    @property
    def label_ar(self) -> str:
        return COCO_NAMES_AR[self.class_id]


class YoloOnnxDetector:
    """COCO-80 object detector. Coordinates returned in source-image pixels."""

    def __init__(
        self,
        model_path: str | Path,
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        max_detections: int = 40,
    ) -> None:
        self.model_path = str(model_path)
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.max_detections = int(max_detections)
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        # Use the CPU backend for broad Android compatibility.
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        cv2.setNumThreads(2)

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] < 3:
            return []
        original_h, original_w = image_bgr.shape[:2]
        if original_h <= 0 or original_w <= 0:
            return []

        inp, ratio, pad_x, pad_y = self._letterbox(image_bgr, self.input_size)
        blob = cv2.dnn.blobFromImage(
            inp, scalefactor=1.0 / 255.0, size=(self.input_size, self.input_size),
            mean=(0, 0, 0), swapRB=True, crop=False,
        )
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        if not outputs:
            return []
        raw = np.asarray(outputs[0])
        rows = self._as_rows(raw)
        if rows.size == 0:
            return []

        boxes_xywh: list[list[float]] = []
        scores: list[float] = []
        class_ids: list[int] = []
        for row in rows:
            # Standard YOLOv8 detect output: 4 box values + 80 class scores.
            if row.size < 5:
                continue
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            if score < self.confidence_threshold or class_id >= len(COCO_NAMES_AR):
                continue
            cx, cy, bw, bh = (float(v) for v in row[:4])
            # Exported YOLOv8 ONNX models normally report coordinates in input pixels.
            # Handle normalized-coordinate exports as well.
            if max(abs(cx), abs(cy), abs(bw), abs(bh)) <= 2.0:
                cx *= self.input_size
                cy *= self.input_size
                bw *= self.input_size
                bh *= self.input_size
            x1 = (cx - bw / 2 - pad_x) / ratio
            y1 = (cy - bh / 2 - pad_y) / ratio
            x2 = (cx + bw / 2 - pad_x) / ratio
            y2 = (cy + bh / 2 - pad_y) / ratio
            x1 = max(0.0, min(float(original_w - 1), x1))
            y1 = max(0.0, min(float(original_h - 1), y1))
            x2 = max(0.0, min(float(original_w - 1), x2))
            y2 = max(0.0, min(float(original_h - 1), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(score)
            class_ids.append(class_id)

        if not boxes_xywh:
            return []
        indices = cv2.dnn.NMSBoxes(
            boxes_xywh, scores, self.confidence_threshold, self.nms_threshold
        )
        selected = np.asarray(indices).reshape(-1).tolist() if len(indices) else []
        selected.sort(key=lambda i: scores[i], reverse=True)
        return [
            Detection(
                x1=boxes_xywh[i][0], y1=boxes_xywh[i][1],
                x2=boxes_xywh[i][0] + boxes_xywh[i][2],
                y2=boxes_xywh[i][1] + boxes_xywh[i][3],
                class_id=class_ids[i], score=scores[i],
            )
            for i in selected[: self.max_detections]
        ]

    @staticmethod
    def _as_rows(output: np.ndarray) -> np.ndarray:
        """Normalize common YOLOv8 ONNX layouts to [anchors, 4 + classes]."""
        arr = np.squeeze(output)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Unexpected model output shape: {output.shape}")
        # Common YOLOv8: [84, 8400]; alternative: [8400, 84].
        if arr.shape[0] <= 128 and arr.shape[1] > arr.shape[0]:
            arr = arr.T
        if arr.shape[1] < 5:
            raise ValueError(f"Unexpected YOLO output shape: {output.shape}")
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _letterbox(image: np.ndarray, size: int) -> tuple[np.ndarray, float, float, float]:
        h, w = image.shape[:2]
        ratio = min(size / w, size / h)
        new_w, new_h = int(round(w * ratio)), int(round(h * ratio))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (size - new_w) / 2.0
        pad_y = (size - new_h) / 2.0
        left = int(round(pad_x - 0.1))
        right = int(round(pad_x + 0.1))
        top = int(round(pad_y - 0.1))
        bottom = int(round(pad_y + 0.1))
        canvas = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return canvas, ratio, float(left), float(top)
