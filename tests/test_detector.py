import unittest

import numpy as np

from detector import YoloOnnxDetector
from labels_ar import COCO_NAMES_AR
from tracker import IoUTracker
from detector import Detection


class DetectorHelpersTest(unittest.TestCase):
    def test_has_80_arabic_names(self):
        self.assertEqual(len(COCO_NAMES_AR), 80)
        self.assertEqual(COCO_NAMES_AR[56], "كرسي")
        self.assertEqual(COCO_NAMES_AR[0], "شخص")

    def test_transposes_yolov8_channel_first_output(self):
        arr = np.zeros((1, 84, 8400), dtype=np.float32)
        rows = YoloOnnxDetector._as_rows(arr)
        self.assertEqual(rows.shape, (8400, 84))

    def test_keeps_anchor_first_layout(self):
        arr = np.zeros((1, 8400, 84), dtype=np.float32)
        rows = YoloOnnxDetector._as_rows(arr)
        self.assertEqual(rows.shape, (8400, 84))

    def test_rejects_unexpected_shape(self):
        with self.assertRaises(ValueError):
            YoloOnnxDetector._as_rows(np.zeros((1, 2, 3, 4), dtype=np.float32))

    def test_tracker_preserves_id_for_moving_object(self):
        tracker = IoUTracker()
        first = Detection(10, 10, 50, 50, 0, 0.9)
        second = Detection(12, 10, 52, 50, 0, 0.88)
        id1 = tracker.update([first], now=1.0)[0].track_id
        id2 = tracker.update([second], now=1.3)[0].track_id
        self.assertEqual(id1, id2)


if __name__ == "__main__":
    unittest.main()
