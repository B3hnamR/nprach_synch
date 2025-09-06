# Project Review — nprach_synch

## Project Overview
این مخزن پیاده‌سازی کامل همگام‌سازی NPRACH برای NB‑IoT را ارائه می‌دهد، با دو رویکرد:
- DeepNSynch (یادگیری عمیق مبتنی بر ResNet با SeparableConv1D)
- NPRACHSynch (خط مبنای تحلیلی مطابق [CHO])
همچنین تولید موج NPRACH (پیکربندی 0)، مدل کانال 3GPP UMi با Sionna، لایهٔ CFO، و نوت‌بوک‌های Train/Evaluate را شامل می‌شود. یک ماژول پیکربندی خودکار زمان‌اجرا (runtime/auto_config.py) برای پیشنهاد تنظیمات ایمن/کارا بسته به منابع سیستم افزوده شده است.

## Architecture & Dependencies
- زبان/پشته: Python، TensorFlow 2.8.x، Sionna 0.13.x، NumPy/Scipy/Matplotlib، Jupyter
- ماژول‌ها و نقاط اتصال:
  - nprach/
    - NPRACHConfig: پارا��ترهای سیگنال NPRACH (پهنای‌باند، SCS=3.75kHz، زمان CP/سمبل، تعداد SG/تکرار، DFT size=48)
    - NPRACHPattern: تولید الگوی پرش فرکانسی و گام‌های پرش با GoldSequence بر اساس TS 36.211
    - NPRACH: تولید سیگنال زمان‌-دامنه و اکسپوز seq_indices و freq_patterns برای گیرنده‌ها
  - e2e/
    - CFO: لایهٔ Keras برای اعمال CFO از ppm به f_off_norm (نرمال به fs) و فازدهی
    - E2E: زنجیرهٔ ارسال→کانال UMi (Sionna)→اعمال تاخیر/CFO→گیرنده (baseline/DL)؛ محاسبهٔ لا‌س‌ها/متریک‌ها برای train/eval
  - synch/
    - NPRACHSynch (baseline): پردازش تفاضلی SGها، ساخت بردار v برحسب گام‌های پرش، FFT روی v، آستانه‌گذاری τ (pfa)، برآورد ToA با درون‌یابی مربعی و CFO از زاویهٔ همبستگی
    - DeepNSynch: ساخت RG، نرمال‌سازی، استخراج REهای پرش‌کرده، شبکهٔ ResNet سبک با سه سر (تشخیص، ToA، CFO)
  - runtime/auto_config.py: شناسایی GPU/CPU/RAM/OS و توصیهٔ batch-size، mixed precision، threads، AUTOTUNE، backend رسم
  - parameters.py: پارامترهای عمومی آموز��/ارزیابی (NPRACH_NUM_SC، BATCH_SIZE_*, MAX_CFO_PPM_TRAIN، ...)
- وابستگی‌ها (requirements.txt):
  - tensorflow==2.8.4
  - sionna==0.13.0 (توجه: README اشاره به 0.14.0 دارد)
  - numpy==1.22.4, scipy==1.8.1, matplotlib==3.5.3, jupyter==1.0.0, ipympl==0.9.3, protobuf==3.19.6, gdown==4.7.1

## Commit History Insights (range examined)
به‌درخواست شما بررسی تاریخچهٔ گیت صرف‌نظر شد. بر اساس CHANGELOG (2025‑09‑02):
- Added: runtime/auto_config.py، scripts/download_weights.py، formulas.md
- Changed: پین نسخه‌ها؛ به‌روزرسانی README/نوت‌بوک‌ها (XLA off به‌صورت پیش‌فرض)
- Fixed: baseline — ایمن‌سازی درون‌یابی ToA و نرمال‌سازی FFT با fft_size

## Issues by Severity (High → Medium → Low)

### High
1) e2e/e2e_system.py — ناسازگاری ابعادی در محاسبهٔ NMSE برای CFO (شاخهٔ eval)
- توضیح: f_off و f_off_est هر دو «نرمال‌شده به فرکانس نمونه‌برداری (fs)» هستند؛ در f_off_err بر پهنای‌باند تقسیم می‌شوند که از نظر ابعاد صحیح نیست و مقیاس‌دهی خطا را تحریف می‌کند.
- کد فعلی (تقریبی):
  - f_off = ppm2Foffnorm(cfo_ppm)  # per fs
  - f_off_err = ((f_off - f_off_est) / config.bandwidth)^2
- راهکار کمینه: تبدیل هر دو به Hz با ضرب در SAMPLING_FREQUENCY و سپس نرمال‌سازی بر bandwidth:
  - f_off_hz = f_off * SAMPLING_FREQUENCY؛ f_est_hz = f_off_est * SAMPLING_FREQUENCY
  - f_off_err = ((f_off_hz - f_est_hz) / config.bandwidth)^2

### Medium
1) synch/baseline.py — ریسک شکست هنگام fft_size < v_len
- موضع: در هر دو مسیر call و _build_detection_threshold، طول zero‑pad به صورت (self._fft_size - v_len) محاسبه و در tf.concat مصرف می‌شود. اگر منفی باشد، concat خطا می‌دهد.
- رفع پیشنهادی: assert ساده قبل از concat یا clamp؛ نمونه: tf.debugging.assert_greater_equal(self._fft_size, v_len).

2) Reproducibility آستانهٔ τ
- τ از نمونه‌گیری نویز تصادفی بدون seed ثابت به‌صورت تجربی استخراج می‌شود؛ این می‌تواند run‑to‑run variance ایجاد کند.
- رفع: افزودن seed اختیاری (یا context manager) + قابلیت cache/بارگذاری τ از فایل.

### Low
1) ناهمخوانی نسخهٔ Sionna بین README (0.14.0) و requirements (0.13.0)
- یکی‌سازی مستندات یا ارتقا/تست نسخهٔ واقعی.

2) e2e/e2e_system.py — sys.path.append('..') غیرضروری
- با imports نسبی داخل پکیج (e2e/__init__.py) نیاز نیست. حذف برای پاک‌سازی مسیر ایمپورت‌ها.

3) Docstring/واحدها برای CFO
- شفاف‌سازی واحدها: "fs‑normalized" در مقابل "Hz" و "bandwidth‑normalized" در baseline/E2E/DeepNSynch برای جلوگیری از ابهام.

4) یادداشت سازگاری ResnetBlock در DeepNSynch
- assert فعلی تضمین همسانی کانال‌ها برای جمع باقیمانده؛ در صورت تغییر ابعاد آینده، 1x1 projection نیاز می‌شود (اکنون مشکلی ندارد).

## Recommendations (refactor/perf/security/tests)
- Correctness
  - اصلاح f_off_err در E2E مطابق پیشنهاد High و همسوسازی docstring‌ها با واحدهای صحیح.
- Robustness
  - افزودن assert برای self._fft_size >= v_len در baseline (هر دو موضع call و threshold builder).
- DX/Docs
  - یکی‌سازی نسخهٔ Sionna بین README/requirements.
  - حذف sys.path.append('..') در e2e/e2e_system.py.
- Reproducibility
  - seed اختیاری برای ساخت τ + cache/بارگذاری τ.
- Tests
  - واحد برای NPRACHPattern (درستی الگوهای پرش)، CFO (ppm→norm→Hz)، baseline τ با pfa مشخص.
  - smoke test سبک CPU‑only (Baseline) + sanity برای خطای CFO با CFO معلوم.

## Proposed Roadmap (incremental)
- Milestone 1 (هفته 1)
  - اصلاح NMSE CFO در E2E، افزودن assert fft_size>=v_len در baseline.
  - به‌روزرسانی اسناد واحدهای CFO در docstring‌ها.
- Milestone 2 (هفته 2)
  - یکی‌سازی نسخهٔ Sionna (تست/تایید 0.13.0 یا ارتقا به 0.14.0)، حذف sys.path.append.
  - افزودن seed اختیاری و cache آستانهٔ τ.
- Milestone 3 (هفته 3)
  - افزودن تست‌های واحد و smoke؛ درصورت موجود بودن، ادغام در CI سبک.

## Suggested Minimal Patches (not applied)
- E2E CFO NMSE (نمایشی):
```diff
--- a/e2e/e2e_system.py
+++ b/e2e/e2e_system.py
@@
-            # CFO NMSE
-            f_off = self.cfo.ppm2Foffnorm(cfo_ppm)
-            f_off_err = tf.where(tx_ue,
-                        tf.square((f_off-f_off_est)/self.config.bandwidth), 0.0)
+            # CFO NMSE (dimensionally correct): convert to Hz then normalize by bandwidth
+            f_off_norm = self.cfo.ppm2Foffnorm(cfo_ppm)          # unitless (per fs)
+            f_off_hz   = f_off_norm * SAMPLING_FREQUENCY         # Hz
+            f_est_hz   = f_off_est   * SAMPLING_FREQUENCY        # Hz
+            denom_hz   = self.config.bandwidth                   # Hz
+            f_off_err  = tf.where(tx_ue,
+                           tf.square((f_off_hz - f_est_hz) / denom_hz), 0.0)
```
- Baseline fft_size guard (نمایشی):
```python
# در synch/baseline.py پیش از ساخت v_freq
v_len = max_hop*2+1
# محافظت از concat منفی
tf.debugging.assert_greater_equal(self._fft_size, v_len,
    message="fft_size must be >= v_len (hop-spectrum length)")
```

---
- Security: مورد خاصی یافت نشد.
- Performance: استفاده از AUTOTUNE در tf.data و محدودسازی threadها در auto_config مناسب است؛ فعال‌سازی آگاهانهٔ XLA فقط در محیط‌های سازگار پیشنهاد می‌شود.
