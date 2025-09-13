# تغییرات (Changelog)

این سند تغییرات اعمال‌شده بر روی پروژه nprach_synch را به‌صورت شفاف، با ذکر دلیل هر تغییر، ثبت می‌کند.

## 2025-09-02

### Added
- runtime/auto_config.py
  - ماژول «پیکربندی خودکار زمان‌اجرا» برای تشخیص منابع سیستم (GPU/CPU/RAM/OS) و توصیه/اعمال تنظیمات امن و کارا (mixed precision روی GPU، خاموش بودن XLA به‌صورت پیش‌فرض، تنظیم threadها، AUTOTUNE برای tf.data، backend رسم inline).
- scripts/download_weights.py
  - اسکریپت کمکی برای دانلود weights.dat به ریشه پروژه (در حال حاضر لینک Google Drive عمومی نیست و ممکن است 403 بدهد؛ در این صورت دانلود دستی یا آموزش محلی لازم است).
- formulas.md
  - نسخهٔ تمیز و قابل کپی از مفاهیم و معادلات مقاله (LaTeX داخل Markdown) جهت استفاده در VS Code و مستندسازی تمرین‌ها.

### Changed
- requirements.txt (پین نسخه‌ها برای بازتولیدپذیری)
  - tensorflow==2.8.4
  - sionna==0.13.0
  - numpy==1.22.4, scipy==1.8.1, matplotlib==3.5.3, jupyter==1.0.0, ipympl==0.9.3
  - protobuf==3.19.6 (سازگار با TF 2.8.x)
  - gdown==4.7.1
  - دلیل: جلوگیری از ناسازگاری‌های نسخه‌ای (به‌ویژه TF/Sionna/protobuf) و حذف وابستگی غیرضروری به Sionna RT/Mitsuba.
- README.md
  - بازنویسی بخش Setup و Quickstart: ساخت venv، نصب وابستگی‌های پین‌شده، نکات Windows/WSL2، غیرفعال بودن XLA به‌صورت پیش‌فرض، توصیه به استفاده از auto_config.
- Evaluate.ipynb و Train.ipynb
  - تغییر @tf.function(jit_compile=True) به jit_compile=False برای پرهیز از خطاهای XLA در محیط‌های CPU-only/Windows/WSL.
  - تغییر plotting به %matplotlib inline برای حذف وابستگی سخت به ipympl.

### Fixed
- synch/baseline.py — درون‌یابی ToA (مرزبندی امن)
  - کلیپ ایندکس‌های k_max±1 به بازه [0, fft_size-1] و افزودن ε کوچک به مخرج برای جلوگیری از تقسیم بر صفر. نتیجه: رفع کرش‌های gather/InvalidArgument روی FFTهای کوچک یا حالات م��زی.
- synch/baseline.py — نرمال‌سازی FFT در ساخت آستانه تشخیص
  - تغییر تقسیم از مقدار ثابت 256 به self._fft_size. نتیجه: آستانه تشخیص منطبق با اندازهٔ واقعی FFT (برای 256 تغییری در رفتار عددی رخ نمی‌دهد).

### Notes
- اجرای پایدار روی WSL2:
  - با فایل C:\Users\Behnam\.wslconfig می‌توان RAM/Swap/CPU اختصاصی WSL2 را افزایش داد (مثلاً memory=20GB, swap=24GB, processors=8) و سپس با `wsl --shutdown` اعمال کرد.
  - در ابتدای نوت‌بوک‌ها محدود کردن threadها می‌تواند پیک مصرف حافظه را کم کند:
    ```python
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    ```
- وزن‌های مدل DL:
  - لینک عمومی فعلی در دسترس نیست (403). برای ارزیابی DL: (الف) آموزش سبک محلی با Train.ipynb (برای تولید weights.dat)، یا (ب) استفاده از ماشین GPU/WSL2-CUDA سازگار برای آموزش کامل.
- ارزیابی Baseline و نمودارها:
  - برای هم‌خوانی با مقاله، در نمودارهای وابسته به CFO ��ز «فیلتر کردن نمونه‌ها با |CFO| نزدیک به مقدار هدف» استفاده شد تا اثر CFO ثابت ایزوله شود (کدهای تمرینی در project practice.md/Notebooks مستند شده‌اند).

---

## ملاحظات سازگاری و اجرای پایدار
- پشتهٔ توصیه‌شده: Python 3.8 + TF 2.8.4 + Sionna 0.13.0 + protobuf 3.19.6 + numpy 1.22.x.
- XLA پیش‌فرض غیرفعال است؛ در محیط‌های سازگار (GPU/CUDA/XLA) می‌توان آن را آگاهانه فعال کرد.
- plotting به‌صورت inline است؛ برای تعامل لازم است ipympl نصب و دستور به widget برگردانده شود.

## نکات مهاجرت (Upgrade Notes)
- اگر قبلاً fft_size ≠ 256 استفاده می‌کردید، دقت کنید آستانهٔ baseline اکنون به‌درستی نرمال شده و ممکن است رفتار عددی نسبت به قبل (اشتباه) اصلاح شود.
- اگر محیط شما XLA را پشتیبانی می‌کند و تمایل دارید، jit_compile=True را مجدداً فعال کنید.

## تست دود پیشنهادی (Smoke Test)
- اجرای baseline سبک روی CPU:
  ```bash
  python - << 'PY'
  from e2e import E2E
  BATCH=16
  sys = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=24, fft_size=256, pfa=0.999)
  out = sys(BATCH, max_cfo_ppm=10., ue_prob=0.5)
  print('OK:', len(out))
  PY
  ```


## 2025-09-13
- Fix: define f_off in E2E eval return path to avoid UnboundLocalError and align CFO NMSE units.
- Docs: Update README (weights generation/verify, Evaluate_prepared.ipynb, smoke_test).
- Add: scripts/smoke_test.py; .gitignore for artifacts; remove stray .docx and old results .res files.

