# بصير — كاشف الأشياء بالكاميرا

تطبيق تجريبي ببايثون يعرض معاينة الكاميرا مباشرة، ويضع صندوقًا واسمًا عربيًا ورقم تتبّع فوق كل جسم يتعرّف عليه النموذج. يطابق متعقب خفيف الصناديق بين الإطارات المتتالية ليحافظ على رقم الجسم ما دام ظاهرًا. الواجهة مبنية بـKivy، والوصول إلى كاميرا Android عبر Camera4Kivy، والاستدلال محليًا باستخدام OpenCV DNN ونموذج ONNX؛ لا يرسل التطبيق إطارات الكاميرا إلى خادم.

## نطاق الكشف

المشروع مهيأ لكشف **80 فئة عامة** من مجموعة COCO، وليس لفئة واحدة. تشمل الفئات الأشخاص والمركبات والحيوانات وأدوات المنزل والطعام والأجهزة الإلكترونية وغيرها، ولكل فئة اسم عربي. تُرفق نسخة صغيرة من نموذج YOLOv8n بصيغة ONNX.

لا يستطيع نموذج محدود الفئات ضمان اكتشاف «أي شيء» حرفيًا. إذا لم تكن الفئة من الفئات التي دُرّب عليها النموذج، فقد لا تظهر أو قد تُصنّف خطأ. مثال مهم: الكرسي المتحرك ليس فئة مستقلة في قائمة COCO القياسية؛ يلزم نموذج مفتوح المفردات أو نموذج مُدرّب/مخصّص ببيانات لهذه الفئة. يمكن استبدال ملف النموذج مستقبلًا بنموذج متوافق مع محلل YOLOv8 في `detector.py`، مع تعديل `labels_ar.py` لتطابق ترتيب الفئات.

## التشغيل على الحاسوب

يتطلب Python 3.10 أو أحدث، كاميرا متصلة، وبيئة سطح مكتب تدعم Kivy.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

على Linux قد تحتاج مكتبات Kivy النظامية وكاميرا متاحة. `camera4kivy` يستخدم مزوّد الكاميرا الملائم للمنصة؛ اختبار كاميرا الحاسوب يعتمد على المزوّد المثبّت.

## بناء APK على Linux/Ubuntu

يلزم تثبيت Java وBuildozer وأدوات Android SDK/NDK المتوافقة. ملف `buildozer.spec` يحدد صلاحية الكاميرا، معمارية ARM، واعتماديات CameraX وOpenCV.

```bash
pip install buildozer
buildozer android debug
```

تتم أول عملية بناء بعد تنزيل أدوات Android وقد تستغرق وقتًا. بعد نجاحها ستجد APK ضمن `bin/`. للسماح بتثبيت APK الاختباري، فعّل التثبيت من هذا المصدر على الهاتف. وافق على صلاحية الكاميرا عند أول تشغيل.

> لم يُبنَ APK في بيئة إعداد هذا المشروع؛ فهي لا تحتوي على Android SDK أو Buildozer. اختبر محرك الكشف آليًا على الحاسوب، واتبع الأوامر أعلاه لإنتاج APK في بيئة بناء Android مجهزة.

## عناصر التحكم

- وجّه الكاميرا إلى المشهد؛ تُحدّث الصناديق والأسماء وأرقام التتبع ونسبة الثقة بالعربية عدة مرات في الثانية لتقليل الحمل على الهاتف.
- زر **إيقاف التحليل** يوقف الاستدلال مؤقتًا، مع بقاء معاينة الكاميرا.
- غيّر `confidence_threshold` في `detector.py` لضبط حساسية الكشف. خفض القيمة يزيد الاكتشافات لكنه قد يزيد النتائج الخاطئة.

## البنية

- `main.py`: واجهة Kivy والكاميرا الحية وطبقة الصناديق/التسميات وصلاحية Android.
- `detector.py`: تحميل ONNX، تجهيز الصورة، فك مخرجات YOLO، وإزالة الصناديق المتكررة.
- `tracker.py`: ربط الصناديق بين الإطارات وإسناد أرقام ثابتة مؤقتًا للأجسام.
- `labels_ar.py`: قاموس أسماء COCO العربية، بنفس ترتيب فئات النموذج.
- `assets/yolov8n.onnx`: نموذج الكشف.
- `camerax_provider/`: إعدادات مزود CameraX المطلوبة لتغليف Kivy على Android.

## ملاحظات الترخيص والمصادر

ملف النموذج مأخوذ من مستودع Unity `inference-engine-yolo` المرخّص GPL-3.0. راجع شروطه قبل إعادة التوزيع أو الاستخدام التجاري. Camera4Kivy ومثال OpenCV المرجعي مرخصان MIT، ومستودع Camera4Kivy مؤرشف؛ قد تحتاج إصدارات Android أو Buildozer المستقبلية إلى تحديث مزوّد الكاميرا.

المصادر الفنية:
- [YOLOv8 ONNX model and model card](https://huggingface.co/unity/inference-engine-yolo)
- [COCO classes and dataset overview](https://docs.ultralytics.com/datasets/detect/coco)
- [Camera4Kivy Android setup and camera analysis](https://github.com/Android-for-Python/Camera4Kivy)
- [python-for-android packaging overview](https://github.com/kivy/python-for-android)
- [Buildozer specifications](https://buildozer.readthedocs.io/en/latest/specifications.html)
- [ONNX Runtime mobile YOLO tutorial](https://onnxruntime.ai/docs/tutorials/mobile/pose-detection.html)
