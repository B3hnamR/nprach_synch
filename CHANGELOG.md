# تغییرات (Changelog)

این سند تغییرات اعمال‌شده بر روی پروژه nprach_synch را به‌صورت شفاف، با ذکر دلیل هر تغییر، ثبت می‌کند.

## 2025-09-02

### Added
- requirements.txt
  - توضیح: نسخه‌های وابستگی‌ها پین شد تا محیط اجرای قابل بازتولید و پایدار فراهم شود.
  - جزئیات: Python 3.8 هدف‌گذاری شده؛ TensorFlow 2.8.4، Sionna 0.14.0، numpy 1.22.4، scipy 1.8.1، matplotlib 3.5.3، jupyter 1.0.0، ipympl 0.9.3، protobuf 3.20.3، gdown 4.7.1.
  - دلیل: در وضعیت قبلی نسخه‌ها پین نبودند و اجرای پروژه روی نسخه‌های جدیدتر باعث خطاهای نصب/API می‌شد (به‌ویژه TF/Sionna/Numpy).

- scripts/download_weights.py
  - توضیح: اسکریپت ساده برای دانلود خودکار weights.dat به ریشه پروژه.
  - دلیل: Evaluate.ipynb بدون weights.dat اجرا نمی‌شود و FileNotFoundError می‌دهد؛ این اسکریپت فرآیند دانلود را ساده و قابل اتکا می‌کند.

### Changed
- README.md
  - توضیح: بخش Setup بازنویسی شد با مراحل مشخص ایجاد محیط مجازی Python 3.8، نصب وابستگی‌های پین‌شده، دانلود وزن‌ها، نکات ویندوز/WSL2، و توضیح غیرفعال‌سازی XLA.
  - دلیل: شفاف‌سازی راه‌اندازی و کاهش خطاهای متداول (XLA، نسخه‌ها، plotting).

- Train.ipynb
  - تغییر: @tf.function(jit_compile=True) به @tf.function(jit_compile=False) تغییر کرد.
  - دلیل: اجرای XLA روی بسیاری از سیستم‌ها (به‌خصوص Windows/CPU-only) با خطای Unimplemented/Unsupported مواجه می‌شود. غیرفعال‌سازی XLA، قابلیت اجرای گسترده‌تر را تضمین می‌کند.

- Evaluate.ipynb
  - تغییرات:
    - سه دکوراتور @tf.function(jit_compile=True) برای توابع sample_sys_snr، sample_sys_cfo، sample_sys_ptx به jit_compile=False تغییر کردند.
    - %matplotlib widget به %matplotlib inline تغییر کرد.
  - دلیل: جلوگیری از خطاهای XLA و حذف وابستگی سخت به ipympl برای نمایش نمودارها در محیط‌هایی که ipympl نصب نیست.

### Fixed
- synch/baseline.py
  - اصلاح: در متد _build_detection_threshold نرمال‌سازی FFT از مقدار ثابت 256 به self._fft_size تغییر کرد:
    - قبلاً: تقسیم بر tf.complex(tf.constant(256, tf.float32), 0.0)
    - اکنون: تقسیم بر tf.complex(tf.constant(self._fft_size, tf.float32), 0.0)
  - دلیل: با fft_size≠256 آستانه تشخیص اشتباه محاسبه می‌شد. این اصلاح سازگاری آستانه با اندازه واقعی FFT را تضمین می‌کند (بدون تغییر رفتار برای مقدار پیش‌فرض 256).

---

## ملاحظات سازگاری و اجرای پایدار
- نسخه‌های پین‌شده در requirements.txt بر مبنای Python 3.8 + TF 2.8 + Sionna 0.14 انتخاب شده‌اند تا با APIهای موجود پروژه سازگار باشند.
- XLA در نوت‌بوک‌ها به‌صورت پیش‌فرض غیرفعال است؛ در صورت داشتن پشته سازگار (GPU/CUDA/XLA) می‌توان آن را دوباره فعال کرد.
- Evaluate.ipynb با %matplotlib inline به جای %matplotlib widget اجرا می‌شود تا وابستگی به ipympl حذف شود. در صورت نیاز به تعامل، ipympl را نصب و دستور را به widget برگردانید.

## نکات مهاجرت (Upgrade Notes)
- baseline: اگر قبلاً از fft_size غیر از 256 استفاده می‌کردید، آستانه تشخیص اکنون به‌درستی با اندازه FFT شما منطبق می‌شود (رفتار عددی ممکن است نسبت به قبل اصلاح شود). برای fft_size=256 تغییری در رفتار ایجاد نمی‌شود.
- نوت‌بوک‌ها: اگر محیط شما XLA را پشتیبانی می‌کند و تمایل به فعال‌سازی دارید، jit_compile=True را مجدداً اعمال کنید.

## تست دود پیشنهادی (Smoke Test)
- برای اطمینان از سازگاری TF/Sionna پیش از اجرای Evaluate روی وزن‌ها، یک اجرای baseline کوچک انجام دهید:
  - اجرای Evaluate.ipynb با سیستم baseline و pfa=0.999 و batch کوچک؛ در صورت ناسازگاری نسخه‌ها، در همین مرحله خطا مشخص می‌شود.

