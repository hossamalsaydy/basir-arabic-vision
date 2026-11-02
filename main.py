"""Arabic live camera object detection app built with Kivy and Camera4Kivy."""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
from kivy.app import App
from kivy.clock import Clock, mainthread
from kivy.core.window import Window
from kivy.graphics import Color, Line, Rectangle, RoundedRectangle
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.utils import platform
from camera4kivy import Preview

from detector import Detection, YoloOnnxDetector
from tracker import IoUTracker

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except ImportError:
    arabic_reshaper = None
    get_display = None

__version__ = "0.1.0"
INFERENCE_INTERVAL = 0.28  # limit CPU load on phones (about 3.5 detections/second)


def arabic_display(text: str) -> str:
    """Shape Arabic glyphs for Kivy's basic text renderer when helpers are present."""
    if arabic_reshaper is None or get_display is None:
        return text
    try:
        return get_display(arabic_reshaper.reshape(text))
    except Exception:
        return text


def arabic_digits(value: int) -> str:
    return str(value).translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))


class TagLabel(Label):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.height = dp(29)
        self.padding = (dp(8), dp(2))
        self.halign = "center"
        self.valign = "middle"
        self.color = (0.03, 0.08, 0.09, 1)
        self.bold = True
        self.font_size = "13sp"
        with self.canvas.before:
            Color(0.38, 0.97, 0.71, 0.96)
            self._background = RoundedRectangle(radius=[dp(7)])
        self.bind(pos=self._sync_background, size=self._sync_background)

    def _sync_background(self, *_args):
        self._background.pos = self.pos
        self._background.size = self.size


class DetectionOverlay(Widget):
    """Draw boxes in Kivy's bottom-left coordinate space and Arabic tag widgets."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tags: list[TagLabel] = []

    def clear(self):
        self.canvas.clear()
        for tag in self._tags:
            self.remove_widget(tag)
        self._tags.clear()

    def show_detections(
        self, detections: list[Detection], frame_size: tuple[int, int],
        image_pos: tuple[float, float], scale: float, mirrored: bool,
    ):
        self.clear()
        frame_w, frame_h = frame_size
        if frame_w <= 0 or frame_h <= 0 or scale <= 0:
            return
        ox, oy = float(image_pos[0]), float(image_pos[1])
        for d in detections:
            x1, x2 = d.x1, d.x2
            if mirrored:
                x1, x2 = frame_w - d.x2, frame_w - d.x1
            left = self.x + ox + x1 * scale
            right = self.x + ox + x2 * scale
            bottom = self.y + oy + (frame_h - d.y2) * scale
            top = self.y + oy + (frame_h - d.y1) * scale
            left = max(self.x, min(self.right, left))
            right = max(self.x, min(self.right, right))
            bottom = max(self.y, min(self.top, bottom))
            top = max(self.y, min(self.top, top))
            if right <= left or top <= bottom:
                continue
            with self.canvas:
                Color(0.12, 0.95, 0.67, 0.95)
                Line(rectangle=(left, bottom, right - left, top - bottom), width=1.8)
            text = f"{d.label_ar} · {arabic_digits(d.track_id)} · {arabic_digits(int(d.score * 100))}٪"
            tag = TagLabel(text=arabic_display(text), width=dp(126))
            tag_size = tag.texture_size[0] if tag.texture_size else dp(126)
            tag.width = min(max(dp(92), tag_size + dp(18)), dp(178))
            tag_x = max(self.x, min(self.right - tag.width, left))
            tag_y = top + dp(2)
            if tag_y + tag.height > self.top:
                tag_y = max(self.y, top - tag.height - dp(2))
            tag.pos = (tag_x, tag_y)
            self.add_widget(tag)
            self._tags.append(tag)


class CameraPreview(Preview):
    def __init__(self, detector: YoloOnnxDetector, **kwargs):
        super().__init__(**kwargs)
        self.detector = detector
        self.tracker = IoUTracker()
        self.last_inference = 0.0
        self.busy = threading.Lock()
        self.panel: CameraPanel | None = None
        self.last_error_reported = False

    def analyze_pixels_callback(self, pixels, image_size, image_pos, scale, mirror):
        now = time.monotonic()
        if self.panel and not self.panel.analysis_enabled:
            return
        if now - self.last_inference < INFERENCE_INTERVAL:
            return
        if not self.busy.acquire(blocking=False):
            return
        self.last_inference = now
        try:
            width, height = int(image_size[0]), int(image_size[1])
            rgba = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 4)
            # Camera4Kivy supplies RGBA bytes. OpenCV DNN takes an image array in BGR order.
            import cv2
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            detections = self.tracker.update(self.detector.detect(bgr), now)
            Clock.schedule_once(
                lambda _dt: self._display(detections, (width, height), image_pos, scale, mirror),
                0,
            )
        except Exception as exc:
            if not self.last_error_reported:
                self.last_error_reported = True
                Clock.schedule_once(lambda _dt, err=str(exc): self._report_error(err), 0)
        finally:
            self.busy.release()

    @mainthread
    def _display(self, detections, frame_size, image_pos, scale, mirror):
        if self.panel:
            self.panel.show(detections, frame_size, image_pos, scale, mirror)

    @mainthread
    def _report_error(self, message: str):
        if self.panel:
            self.panel.set_status("تعذّر تحليل الصورة؛ راجع سجل التشغيل")
            print(f"Detection error: {message}")


class CameraPanel(FloatLayout):
    def __init__(self, detector, **kwargs):
        super().__init__(**kwargs)
        self.preview = CameraPreview(detector, aspect_ratio="16:9")
        self.preview.panel = self
        self.overlay = DetectionOverlay()
        self.add_widget(self.preview)
        self.add_widget(self.overlay)
        self.status_label: Label | None = None
        self.analysis_enabled = True

    def on_size(self, *_args):
        # Keep the Preview and overlay aligned exactly within the camera panel.
        if hasattr(self, "preview"):
            self.preview.pos = self.pos
            self.preview.size = self.size
            self.overlay.pos = self.pos
            self.overlay.size = self.size

    def show(self, detections, frame_size, image_pos, scale, mirror):
        self.overlay.show_detections(detections, frame_size, image_pos, scale, mirror)
        if self.status_label:
            self.status_label.text = arabic_display(f"تم رصد {arabic_digits(len(detections))} عناصر")

    def set_status(self, text: str):
        if self.status_label:
            self.status_label.text = arabic_display(text)


class Header(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", padding=(dp(20), dp(10)), spacing=dp(2), **kwargs)
        with self.canvas.before:
            Color(0.035, 0.075, 0.11, 1)
            self.bg = Rectangle(pos=self.pos, size=self.size)
        self.bind(pos=self._sync_bg, size=self._sync_bg)
        self.add_widget(Label(
            text=arabic_display("بَصير"), font_size="22sp", bold=True,
            color=(0.37, 0.98, 0.70, 1), halign="right", valign="middle",
        ))
        self.add_widget(Label(
            text=arabic_display("رؤية مباشرة · كشف ذكي للأشياء"), font_size="12sp",
            color=(0.77, 0.84, 0.88, 1), halign="right", valign="middle",
        ))

    def _sync_bg(self, *_args):
        self.bg.pos, self.bg.size = self.pos, self.size


class MainView(BoxLayout):
    def __init__(self, detector, **kwargs):
        super().__init__(orientation="vertical", spacing=0, **kwargs)
        self.header = Header(size_hint_y=None, height=dp(84))
        self.camera_panel = CameraPanel(detector)
        self.footer = BoxLayout(
            orientation="horizontal", size_hint_y=None, height=dp(72),
            padding=(dp(14), dp(10)), spacing=dp(12),
        )
        with self.footer.canvas.before:
            Color(0.035, 0.075, 0.11, 1)
            self.footer_bg = Rectangle(pos=self.footer.pos, size=self.footer.size)
        self.footer.bind(pos=self._sync_footer_bg, size=self._sync_footer_bg)
        self.status = Label(
            text=arabic_display("الكاميرا جاهزة"), color=(0.88, 0.92, 0.95, 1),
            halign="right", valign="middle", font_size="14sp",
        )
        self.camera_panel.status_label = self.status
        self.toggle_button = Button(
            text=arabic_display("إيقاف التحليل"), size_hint_x=None, width=dp(142),
            background_normal="", background_color=(0.20, 0.69, 0.51, 1),
            color=(0.02, 0.08, 0.08, 1), bold=True,
        )
        self.toggle_button.bind(on_release=self.toggle_analysis)
        self.analysis_enabled = True
        self.footer.add_widget(self.status)
        self.footer.add_widget(self.toggle_button)
        self.add_widget(self.header)
        self.add_widget(self.camera_panel)
        self.add_widget(self.footer)

    def _sync_footer_bg(self, *_args):
        self.footer_bg.pos, self.footer_bg.size = self.footer.pos, self.footer.size

    def toggle_analysis(self, *_args):
        preview = self.camera_panel.preview
        self.analysis_enabled = not self.analysis_enabled
        self.camera_panel.analysis_enabled = self.analysis_enabled
        if self.analysis_enabled:
            self.toggle_button.text = arabic_display("إيقاف التحليل")
            self.camera_panel.set_status("التحليل يعمل")
            preview.last_inference = 0.0
        else:
            self.toggle_button.text = arabic_display("استئناف التحليل")
            self.camera_panel.overlay.clear()
            self.camera_panel.set_status("التحليل متوقف")


class ArabicVisionApp(App):
    title = "بصير - كاشف الأشياء"

    def build(self):
        if Window is not None:
            Window.clearcolor = (0.03, 0.06, 0.09, 1)
        model_path = Path(__file__).resolve().parent / "assets" / "yolov8n.onnx"
        self.detector = YoloOnnxDetector(model_path)
        self.root_view = MainView(self.detector)
        return self.root_view

    def on_start(self):
        if platform == "android":
            from android.permissions import Permission, request_permissions
            request_permissions([Permission.CAMERA], self._camera_permission_result)
        else:
            Clock.schedule_once(self._connect_camera, 0.25)

    def _camera_permission_result(self, permissions, grants):
        if grants and all(grants):
            Clock.schedule_once(self._connect_camera, 0.25)
        else:
            self.root_view.camera_panel.set_status("يرجى السماح باستخدام الكاميرا")

    def _connect_camera(self, _dt):
        try:
            self.root_view.camera_panel.preview.connect_camera(
                analyze_pixels_resolution=640,
                enable_analyze_pixels=True,
                enable_video=False,
            )
            self.root_view.camera_panel.set_status("وجّه الكاميرا إلى الأشياء")
        except Exception as exc:
            self.root_view.camera_panel.set_status("تعذّر تشغيل الكاميرا")
            print(f"Camera error: {exc}")

    def on_stop(self):
        if hasattr(self, "root_view"):
            try:
                self.root_view.camera_panel.preview.disconnect_camera()
            except Exception:
                pass


if __name__ == "__main__":
    ArabicVisionApp().run()
