[app]
title = بصير - كاشف الأشياء
package.name = basirvision
package.domain = org.example
source.dir = .
source.include_exts = py,png,jpg,kv,onnx,ttf,txt,md
source.exclude_dirs = tests,bin,.git,__pycache__
version = 0.1.0
requirements = python3,kivy,camera4kivy,gestures4kivy,opencv,arabic-reshaper,python-bidi
orientation = portrait
fullscreen = 0

# Camera4Kivy CameraX provider for Android.
p4a.hook = camerax_provider/gradle_options.py
android.api = 33
android.minapi = 23
android.archs = arm64-v8a, armeabi-v7a
android.permissions = CAMERA
android.enable_androidx = True

[buildozer]
log_level = 2
warn_on_root = 1
