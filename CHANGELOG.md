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

- runtime/auto_config.py
  - توضیح: ماژول «پیکربندی خودکار زمان‌اجرا» که منابع سیستم (GPU/CPU/RAM/OS) را تشخیص داده و تنظیمات ایمن و کارا برای اجرای TensorFlow و کد پروژه پیشنهاد/اعمال می‌کند. هدف این است که روی طیف گسترده‌ای از ماشین‌ها (از CPU-only تا GPUهای قوی) بدون خطا و با بهره‌وری مناسب اجرا شود.
  - قابلیت‌ها:
    - تشخیص سیستم: GPUهای موجود (نام‌ها)، تعداد CPU، میزان RAM، سیستم‌عامل، نسخه TF (با psutil اختیاری و بهترین تلاش).
    - توصیه تنظیمات: use_gpu، mixed_precision (روی GPU فعال)، jit_compile (پیش‌فرض غیرفعال برای سازگاری)، batch_size_train/eval (مطابق GPU/CPU/RAM)، inter_op_threads/intra_op_threads (براساس CPU)، موازی‌سازی/پریفچ tf.data با AUTOTUNE، و backend مناسب matplotlib (inline).
    - اعمال تنظیمات: تنظیم تعداد threadهای TF، فعال‌سازی memory growth برای GPU، ست‌کردن mixed_float16 در صورت فعال بودن mixed precision.
    - خروجی و خلاصه: تبدیل تنظیمات به dict و چاپ خلاصهٔ مشخصات سیستم + تنظیمات پیشنهادی با summarize().
  - منطق تصمیم‌گیری (Heuristics):
    - اگر GPU موجود باشد: mixed_precision=True، jit_compile=False (پیش‌فرض سازگار)، batch size ~64 برای train/eval (قابل تغییر)، threadها محافظه‌کارانه.
    - اگر سیستم CPU-only یا RAM کم (<8GB) باشد: mixed_precision=False، jit_compile=False، batch size کوچک‌تر (مثلاً 4–16)، threadها محافظه‌کارانه.
    - همیشه tf.data.AUTOTUNE برای parallel_calls و prefetch و ترجیح inline برای plotting جهت حذف وابستگی سخت به ipympl.
  - نحوه استفاده (الگوی بالا-دستی):
    - وارد کردن و گرفتن پروفایل + توصیه:
      ```python
      from runtime.auto_config import (
          get_system_profile, recommend_settings, apply_tf_settings, summarize
      )

      prof = get_system_profile()
      rec  = recommend_settings(prof, mode='eval')  # یا 'train'
      apply_tf_settings(rec)
      print(summarize(prof, rec))

      # به‌صورت اختیاری: اعمال batch size پیشنهادی
      BATCH_SIZE_TRAIN = rec.batch_size_train
      BATCH_SIZE_EVAL  = rec.batch_size_eval

      # به‌صورت اختیاری: استفاده از فلگ XLA پیشنهادشده
      USE_XLA = rec.jit_compile
      ```
    - در نوت‌بوک‌ها: سلول ابتدایی را به این صورت اضافه کنید و jit_compile دکوراتورها را با USE_XLA گره بزنید:
      ```python
      @tf.function(jit_compile=USE_XLA)
      def sample_sys_snr(...):
          ...
      ```
  - نکات سازگاری:
    - XLA در این ماژول به‌صورت پیش‌فرض فعال نمی‌شود تا روی سیستم‌های بدون XLA (Windows/CPU-only) خطا ندهد. کاربران در محیط‌های سازگار می‌توانند USE_XLA را True کنند.
    - psutil اختیاری است؛ در صورت نبود، تشخیص RAM به‌صورت best-effort نادیده گرفته می‌شود.
  - مزایا:
    - پایداری بالا در سیستم‌های ضعیف و قابل‌اجرا بودن بدون خطا (عدم نیاز به XLA، کنترل batch size، threadها).
    - بهره‌وری بهتر روی GPU (mixed precision، memory growth، AUTOTUNE).
    - یک نقطهٔ واحد برای مشاهدهٔ توان سیستم و تنظیمات اعمال‌شده (summarize()).
  - توسعه‌های آینده (پیشنهادی):
    - پروفایل‌های از پیش‌تعریف‌شده مانند 'cpu_safe'، 'gpu_fast' یا تشخیص کلاس GPU برای مقیاس‌دهی پویا.
    - پشتیبانی از ENV (مثلاً NPRACH_PROFILE=gpu_fast) برای override در اجرای headless.
    - لاگ کردن تنظیمات اعمال‌شده در خروجی/فایل و ادغام با CLI (run_train.py/run_eval.py).

### Changed
- requirements.txt
  - تغییر: sionna از 0.14.0 به 0.13.0 و protobuf از 3.20.3 به 3.19.6 به‌روزرسانی شد تا با TensorFlow 2.8.4 سازگار باشد و از وارد شدن ساب‌ماژول RT (وابسته به Mitsuba) جلوگیری شود.
  - دلیل: نسخه 0.14.0 ساینّا هنگام import، rt را هم بارگذاری و به Mitsuba متکی است؛ در این پروژه به RT نیازی نیست و نصب Mitsuba در بسیاری محیط‌ها دردسرساز می‌شود. همچنین TF 2.8.4 با protobuf < 3.20 سازگار است.
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
- synch/baseline.py (ToA interpolation)
  - اصلاح: کلیپ ایندکس‌های k_max±1 به بازه معتبر [0, fft_size-1] و افزودن eps به مخرج درون‌یابی درجه دوم برای جلوگیری از تقسیم بر صفر.
  - دلیل: در اندازه‌های FFT کوچک/مرزی یا برخی ورودی‌ها، gather با ایندکس‌های -1 یا fft_size کرش می‌کرد (InvalidArgument). این اصلاح پایداری روش baseline را افزایش می‌دهد.
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

