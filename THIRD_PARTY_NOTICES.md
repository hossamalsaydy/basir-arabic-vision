# إشعارات تراخيص الأطراف الثالثة

هذا المستودع يحتوي على مصدر المشروع الخاص، إضافةً إلى الملفات الخارجية التالية:

- `assets/yolov8n.onnx`: ملف نموذج مأخوذ من [Unity inference-engine-yolo](https://huggingface.co/unity/inference-engine-yolo)، وموسوم بترخيص GPL-3.0 على صفحة المستودع. انظر [نص GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html). قد تترتب على إعادة توزيع النموذج أو دمجه التزامات الترخيص؛ راجع النص القانوني قبل الاستخدام التجاري أو التوزيع.
- `camerax_provider/`: مزود CameraX من [Android-for-Python/camerax_provider](https://github.com/Android-for-Python/camerax_provider)، بترخيص MIT. عُدّل ملف `gradle_options.py` هنا لحذف إضافة صلاحية الميكروفون لأن التطبيق لا يسجل الصوت.
- `Camera4Kivy`: حزمة Python منشورة بترخيص MIT؛ إعدادها موثق في [المستودع الرسمي](https://github.com/Android-for-Python/Camera4Kivy). المستودع مؤرشف منذ 2023.

أما ملفات المصدر التي أُنشئت لهذا المشروع، فهي مرخصة وفق MIT كما في `LICENSE-CODE.txt`. إشعارات الأطراف الثالثة أعلاه مستقلة عن ترخيص هذه الملفات.
