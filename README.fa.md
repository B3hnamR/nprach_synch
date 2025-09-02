# همگام‌سازی مبتنی بر یادگیری عمیق برای NB-IoT (NPRACH)

این مخزن پیاده‌سازی دو روش همگام‌سازی برای کانال NPRACH در NB-IoT را فراهم می‌کند:
- روش مبتنی بر یادگیری عمیق (DeepNSynch)
- روش پایه‌ی تحلیلی قوی (NPRACHSynch)

همچنین شبیه‌سازی انتها-به-انتها شامل تولید موج NPRACH، مدل کانال 3GPP UMi (با Sionna) و همگام‌سازی/ارزیابی را فراهم می‌کند.

[English README](./README.md)

## فهرست مطالب
- مرور کلی
- ویژگی‌ها
- نیازمندی‌ها
- شروع سریع
- پیکربندی خودکار زمان‌اجرا (جدید)
- ساختار پروژه
- نوت‌بوک‌ها: آموزش و ارزیابی
- نکات CPU/GPU، XLA و رسم نمودار
- تست دود (فقط baseline)
- رفع اشکال
- منابع
- کرِدیت‌ها و مجوز

## مرور کلی
در این کار، الگوریتمی مبتنی بر شبکه‌های عصبی برای تشخیص دستگاه و برآورد زمان رسیدن (ToA) و آفست فرکانسی حامل (CFO) برای NPRACH ارائه می‌شود. معماری شبکه از رزنِت و دانش ساختار پیشامبل 5G NR بهره می‌برد و در برابر یک روش پایه‌ی قوی ارزیابی می‌شود.

## ویژگی‌ها
- پیاده‌سازی موج NPRACH (پیکربندی پیشامبل 0)
- شبیه‌سازی انتها-به-انتها با کانال UMi (Sionna)
- دو روش همگام‌سازی: یادگیری عمیق و پایه
- محیط قابل بازتولید با نسخه‌های پین‌شده
- جدید: پیکربندی خودکار زمان‌اجرا متناسب با منابع سیستم (CPU-only تا GPU قوی)

## نیازمندی‌ها
پیشنهاد می‌شود از Ubuntu 20.04 (یا WSL2 روی ویندوز)، Python 3.8 و TensorFlow 2.8 استفاده کنید.

وابستگی‌ها در `requirements.txt` پین شده‌اند:
- tensorflow==2.8.4
- sionna==0.14.0
- numpy==1.22.4
- scipy==1.8.1
- matplotlib==3.5.3
- jupyter==1.0.0
- ipympl==0.9.3 (اختیاری برای نمودار تعاملی)
- protobuf==3.20.3
- gdown==4.7.1 (برای اسکریپت دانلود وزن‌ها)

## شروع سریع
1) ساخت و فعال‌سازی محیط مجاز�� (Python 3.8):
```
# لینوکس/WSL
python3.8 -m venv .venv && source .venv/bin/activate

# ویندوز PowerShell
py -3.8 -m venv .venv && .venv\Scripts\activate
```

2) نصب وابستگی‌ها:
```
pip install --upgrade pip
pip install -r requirements.txt
```

3) (اختیاری، برای ارزیابی DeepNSynch) دانلود وزن‌ها:
```
python scripts/download_weights.py
# فایل weights.dat در ریشه‌ی پروژه قرار می‌گیرد
```

4) اجرای Jupyter:
```
jupyter notebook
```
سپس `Evaluate.ipynb` یا `Train.ipynb` را باز کنید.

## پیکربندی خودکار زمان‌اجرا (جدید)
ماژول `runtime/auto_config.py` منابع سیستم (GPU/CPU/RAM/OS) را تشخیص می‌دهد و تنظیمات ایمن/بهینه مثل اندازه batch، mixed precision، نخ‌های TF، AUTOTUNE برای tf.data و backend رسم را پیشنهاد/اعمال می‌کند.

نمونه استفاده (در ابتدای نوت‌بوک):
```
from runtime.auto_config import get_system_profile, recommend_settings, apply_tf_settings, summarize

prof = get_system_profile()
rec  = recommend_settings(prof, mode='eval')  # یا 'train'
apply_tf_settings(rec)
print(summarize(prof, rec))

# به‌صورت اختیاری، این مقادیر را در کد خود اعمال کنید
BATCH_SIZE_TRAIN = rec.batch_size_train
BATCH_SIZE_EVAL  = rec.batch_size_eval
USE_XLA = rec.jit_compile  # پیش‌فرض False برای سازگاری گسترده
```
اتصال دکوراتورهای tf.function به فلگ پیشنهادی:
```
@tf.function(jit_compile=USE_XLA)
def sample_sys_snr(...):
    ...
```
منطق کلی:
- GPU: mixed precision روشن، XLA خاموش (پیش‌فرض)، batch ~64، نخ‌ها محافظه‌کارانه
- فقط CPU/رم کم: mixed precision خاموش، XLA خاموش، batch کوچک‌تر (۴–۱۶)
- tf.data: استفاده از AUTOTUNE برای parallel_calls و prefetch
- رسم: backend به صورت inline برای حذف وابستگی سخت به ipympl

## ساختار پروژه
- `nprach/`: پیاده‌سازی موج NPRACH
- `synch/`: روش‌های همگام‌سازی (DeepNSynch, NPRACHSynch)
- `e2e/`: مدل انتها-به-انتها (تولید موج، کانال، همگام‌سازی)
- `runtime/`: ابزارهای زمان‌اجرا
  - `auto_config.py`: تشخیص/توصیه تنظیمات (جدید)
- `parameters.py`: پارامترهای سراسری
- `results/`: نتایج خروجی ارزیابی
- `Train.ipynb`: آموزش DeepNSynch
- `Evaluate.ipynb`: ارزیابی DeepNSynch و baseline و بازتولید نمودارهای مقاله
- `scripts/download_weights.py`: دانلود وزن‌های مدل
- `CHANGELOG.md`: ثبت تغییرات با دلایل

## نوت‌بوک‌ها: آموزش و ارزیابی
- آموزش (`Train.ipynb`):
  - `jit_compile=False` به‌صورت پیش‌فرض برای پایداری؛ فقط در صورت سازگاری TF/XLA/CUDA آن را فعال کنید.
  - از `runtime/auto_config.py` برای تعیین batch size و نخ‌ها استفاده کنید.
- ارزیابی (`Evaluate.ipynb`):
  - برای ارزیابی مدل یادگیری عمیق نیاز به `weights.dat` در ریشه پروژه دارید.
  - baseline بدون وزن نیز قابل اجراست.
  - رسم به‌صورت inline است؛ برای ابزارک‌های تعاملی، `ipympl` نصب و به `%matplotlib widget` برگردید.

## نکات CPU/GPU، XLA و رسم
- XLA پیش‌فرض غیرفعال است (برای سازگاری با Windows/CPU-only).
- در ویندوز، WSL2 (Ubuntu 20.04) توصیه می‌شود؛ در حالت Windows native از TF 2.8 CPU-only استفاده کنید.
- اگر `ipympl` نصب نیست، به صورت `%matplotlib inline` اجرا می‌شود.

فعالسازی XLA در محیط سازگار:
```
USE_XLA = True
@tf.function(jit_compile=USE_XLA)
def my_fn(...):
    ...
```

## تست دود (baseline)
برای اطمینان از سازگاری TF/Sionna بدون نیاز به وزن‌ها:
- سیستم baseline (`E2E('baseline', False, ...)`) با `pfa=0.999`
- batch کوچک (`BATCH_SIZE_EVAL`) و `max_cfo_ppm=10., ue_prob=0.5`
در صورت ناسازگاری نسخه‌ها، خطا در همین مرحله مشخص می‌شود.

## رفع اشکال
- FileNotFoundError برای `weights.dat`:
  - `python scripts/download_weights.py` را اجرا کنید و وجود فایل در ریشه پروژه را بررسی کنید.
- خطا یا ناسازگاری Sionna:
  - از نسخه‌های پین‌شده (`sionna==0.14.0` با `tensorflow==2.8.4`) استفاده کنید.
- خطاهای XLA (Unimplemented/Unsupported):
  - `jit_compile=False` را نگه دارید (پیش‌فرض).
- خطای matplotlib widget:
  - `ipympl` را نصب کنید یا از inline استفاده کنید.
- عدم شناسایی GPU:
  - نصب درست TF GPU/CUDA/cuDNN/Driver را بررسی کنید یا به حالت CPU-only بروید.

## منابع
[A] F. Aït Aoudia, J. Hoydis, S. Cammerer, M. Van Keirsbilck, and A. Keller, "Deep Learning-Based Synchronization for Uplink NB-IoT", 2022. https://arxiv.org/abs/2205.10805

[B] H. Chougrani, S. Kisseleff and S. Chatzinotas, "Efficient Preamble Detection and Time-of-Arrival Estimation for Single-Tone Frequency Hopping Random Access in NB-IoT," IEEE IoT Journal, 8(9):7437-7449, 2021. https://ieeexplore.ieee.org/abstract/document/9263250/

## کرِدیت‌ها
- نگه‌داری، به‌روزرسانی و مستندسازی: Behnam
- پیاده‌سازی اولیه و ارجاعات طبق منابع فوق

## مجوز
© 2022, NVIDIA Corporation. تمامی حقوق محفوظ است.

این کار تحت [مجوز Nvidia](./LICENSE.txt) منتشر شده است.
