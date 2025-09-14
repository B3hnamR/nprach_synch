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
# همگام‌سازی مبتنی بر یادگیری عمیق برای پیوند بالای NB‑IoT (NPRACH)

[English README](./README.md)

این مخزن پیاده‌سازی دو روش همگام‌سازی برای کانال تصادمی باریک‌باند (NPRACH) در NB‑IoT را ارائه می‌کند:
- DeepNSynch (یادگیری عمیق)
- NPRACHSynch (روش تحلیلی مبنا)

همچنین شبیه‌سازی انتها‑به‑انتها شامل تولید موج NPRACH، مدل کانال 3GPP UMi (با Sionna)، و ارزیابی/رسم نتایج را فراهم می‌کند.

## فهرست
- مرور کلی
- ویژگی‌ها
- پیش‌نیازها
- شروع سریع
- پیکربندی خودکار اجرا (جدید)
- ساختار پروژه
- نوت‌بوک‌ها: آموزش و ارزیابی
- نکات CPU/GPU، XLA و رسم
- آزمون دود (فقط مبنا)
- عیب‌یابی
- منابع
- اعتبار و مجوز

## مرور کلی
یک الگوریتم مبتنی بر شبکه عصبی برای تشخیص دستگاه و برآورد زمان‌رسیدن (ToA) و انحراف فرکانس حامل (CFO) برای NPRACH ارائه می‌کنیم. معماری NN از ResNet و دانش ساختار پیش‌کلام استفاده می‌کند و با یک روش تحلیلی قوی مقایسه می‌شود.

## ویژگی‌ها
- پیاده‌سازی موج NPRACH (پیکربندی پیش‌کلام 0)
- شبیه‌سازی انتها‑به‑انتها با کانال UMi (Sionna)
- دو روش همگام‌سازی: یادگیری عمیق و مبنا
- محیط قابل بازتولید با وابستگی‌های پین‌شده
- جدید: پیکربندی خودکار اجرا (batch، mixed precision، نخ‌های TF، AUTOTUNE)

## پیش‌نیازها
پیشنهاد می‌شود از Ubuntu 20.04 (یا WSL2 در ویندوز)، Python 3.8 و TensorFlow 2.8 استفاده کنید.

نسخه‌های پین‌شده در `requirements.txt` (برای Python 3.8) آمده‌اند:
- tensorflow==2.8.4
- sionna==0.13.0
- numpy==1.22.4
- scipy==1.8.1
- matplotlib==3.5.3
- jupyter==1.0.0
- ipympl==0.9.3 (اختیاری برای نمودار تعاملی)
- protobuf==3.19.6
- gdown==4.7.1 (اختیاری؛ در صورت تولید محلی وزن‌ها لازم نیست)

## شروع سریع
1) ساخت و فعال‌سازی محیط مجازی (Python 3.8):
```
# Linux/WSL
python3.8 -m venv .venv && source .venv/bin/activate

# Windows PowerShell
py -3.8 -m venv .venv && .venv\Scripts\activate
```

2) نصب وابستگی‌ها:
```
pip install --upgrade pip
pip install -r requirements.txt
```

3) (اختیاری، برای DeepNSynch) آماده‌سازی وزن‌ها:
```
# گزینه A: فایل وزن خود را در ریشه پروژه با نام weights.dat قرار دهید
# گزینه B: وزن‌های سازگار از نظر شکل را محلی بسازید و تأیید کنید
python scripts/generate_weights.py
python scripts/verify_weights.py weights.dat
```

4) اجرای Jupyter:
```
jupyter notebook
```
نوت‌بوک‌های `Evaluate.ipynb` یا `Train.ipynb` را باز کنید.

## پیکربندی خودکار اجرا (جدید)
فایل `runtime/auto_config.py` منابع سیستم (GPU/CPU/RAM/OS) را تشخیص می‌دهد و تنظیمات امن/کارا پیشنهاد می‌دهد: اندازه batch، mixed precision، نخ‌های TF، AUTOTUNE برای tf.data و backend رسم.

در ابتدای نوت‌بوک اضافه کنید:
```
from runtime.auto_config import get_system_profile, recommend_settings, apply_tf_settings, summarize

prof = get_system_profile()
rec  = recommend_settings(prof, mode='eval')  # یا 'train'
apply_tf_settings(rec)
print(summarize(prof, rec))

BATCH_SIZE_TRAIN = rec.batch_size_train
BATCH_SIZE_EVAL  = rec.batch_size_eval
USE_XLA = rec.jit_compile  # پیش‌فرض False برای سازگاری بیشتر
```
در صورت نیاز:
```
@tf.function(jit_compile=USE_XLA)
def sample_fn(...):
    ...
```

## ساختار پروژه
- `nprach/`: پیاده‌سازی موج NPRACH
- `synch/`: الگوریتم‌های همگام‌سازی (DeepNSynch, NPRACHSynch)
- `e2e/`: شبیه‌سازی انتها‑به‑انتها (تولید، کانال، همگام‌سازی)
- `runtime/`: ابزارهای زمان اجرا
  - `auto_config.py`: تشخیص/توصیه تنظیمات اجرا
- `parameters.py`: پارامترهای کلی (batch، بازه CFO، …)
- `results/`: خروجی‌های تولیدشده توسط Evaluate.ipynb
- `Train.ipynb`: آموزش DeepNSynch
- `Evaluate.ipynb`: مقایسه DeepNSynch و مبنا و بازتولید نمودارها
- `scripts/`: ابزار وزن‌ها
  - `generate_weights.py`: ساخت وزن‌ها در فرمت‌های `.dat/.npz/.h5`
  - `verify_weights.py`: بارگذاری وزن‌ها و اجرای forward آزمایشی
  - `train_deepnsynch.py`: warm‑build و ذخیره وزن‌ها (قابل توسعه برای آموزش واقعی)

## نوت‌بوک‌ها: آموزش و ارزیابی
- آموزش (`Train.ipynb`):
  - `jit_compile=False` برای پایداری؛ فقط در صورت سازگاری TF/XLA/CUDA فعال کنید.
  - از `runtime/auto_config.py` برای انتخاب batch و نخ‌ها استفاده کنید.
- ارزیابی (`Evaluate.ipynb`):
  - برای ارزیابی مدل عمیق، `weights.dat` را در ریشه پروژه قرار دهید (یا محلی بسازید).
  - روش مبنا بدون وزن هم اجرا می‌شود.
  - رسم به‌صورت inline است؛ برای ابزارهای تعاملی `ipympl` نصب و `%matplotlib widget` استفاده کنید.

## نکات CPU/GPU، XLA و رسم
- XLA به‌صورت پیش‌فرض غیرفعال است تا سازگاری حداکثری داشته باشد (در ویندوز/CPU‑only ممکن است خطا دهد).
- در ویندوز، WSL2 (اوبونتو 20.04) پیشنهاد می‌شود؛ در ویندوز بومی از TF 2.8 نسخه CPU‑only و XLA خاموش استفاده کنید.
- در نبود `ipympl` از `%matplotlib inline` استفاده می‌شود.

## آزمون دود (فقط مبنا)
برای آزمون سریع سازگاری TF/Sionna بدون وزن‌ها از مسیر مبنا در `Evaluate.ipynb` استفاده کنید، یا کد زیر را اجرا کنید:
```
from e2e import E2E
BATCH=16
sys = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=24, fft_size=256, pfa=0.999)
out = sys(BATCH, max_cfo_ppm=10., ue_prob=0.5)
print('OK:', len(out))
```

## عیب‌یابی
- FileNotFoundError: `weights.dat`
  - وزن‌های خود را در ریشه قرار دهید یا `python scripts/generate_weights.py` اجرا و سپس با `python scripts/verify_weights.py weights.dat` صحت‌سنجی کنید.
- خطای وارد کردن Sionna یا ناسازگاری API
  - از نسخه‌های پین‌شده `sionna==0.13.0` و `tensorflow==2.8.4` استفاده کنید.
- خطاهای XLA (Unsupported/Unimplemented)
  - `jit_compile=False` را نگه دارید (پیش‌فرض).
- خطای matplotlib widget
  - `ipympl` را نصب کنید یا روی `%matplotlib inline` بمانید.
- عدم شناسایی GPU
  - نصب درست TF/CUDA/cuDNN/Driver را بررسی کنید؛ در غیر این‌صورت در حالت CPU اجرا کنید.

## منابع
[A] F. Aoudia, J. Hoydis, S. Cammerer, M. Van Keirsbilck, and A. Keller, "Deep Learning‑Based Synchronization for Uplink NB‑IoT", 2022. https://arxiv.org/abs/2205.10805

[B] H. Chougrani, S. Kisseleff and S. Chatzinotas, "Efficient Preamble Detection and Time‑of‑Arrival Estimation for Single‑Tone Frequency Hopping Random Access in NB‑IoT," IEEE IoT Journal, 8(9):7437‑7449, 2021. https://ieeexplore.ieee.org/abstract/document/9263250/

## اعتبار و مجوز
- نگه‌داری پروژه، به‌روزرسانی‌ها و مستندسازی: Behnam
- پیاده‌سازی اصلی و مراجع در بالا آمده است.

این کار تحت [مجوز Nvidia](LICENSE.txt) منتشر می‌شود.
# راهنمای پروژه NB‑IoT NPRACH (فارسی و خودمونی)

<div dir="rtl" style="text-align:right">

## خلاصه
این ریپو دو مسیر همگام‌سازی برای سیگنال شروع اتصال NB‑IoT (NPRACH) دارد:
- Baseline (تحلیلی/کلاسیک) → بدون نیاز به وزن (weight) و همین حالا قابل اجرا.
- DeepNSynch (یادگیری عمیق) → برای نتایج واقعی نیاز به weight آموزش‌داده.

ما شبیه‌سازی انتها‑به‑انتها داریم: تولید NPRACH، کانال 3GPP UMi (با Sionna)، و ارزیابی/رسم خروجی‌ها.

نکته مهم: چون وزن‌های مقاله در دسترس نیست، یک weight «سازگار/آزمایشی» می‌توان ساخت تا فقط مدل DL اجرا شود؛ ولی نمودارهای DL با این وزن‌ها قابل استناد نیستند. برای خروجی تحویلی و مطمئن، از Baseline استفاده کنید.

## ویژگی‌ها
- پیاده‌سازی موج NPRACH (پیکربندی 0)
- شبیه‌سازی UMi با Sionna
- دو روش همگام‌سازی: Baseline و DeepNSynch
- فایل‌های نوت‌بوک برای آموزش/ارزیابی + اسکریپت‌های کمکی

## پیش‌نیازها
- Ubuntu 20.04/22.04 یا WSL2 (روی ویندوز)
- Python 3.8
- GPU اختیاری (CPU هم جواب می‌دهد؛ فقط کندتر است)

وابستگی‌های پین‌شده (در `requirements.txt`):
- tensorflow==2.8.4
- sionna==0.13.0
- numpy==1.22.4, scipy==1.8.1, matplotlib==3.5.3, jupyter==1.0.0, ipympl==0.9.3
- protobuf==3.19.6
- gdown==4.7.1 (اختیاری)

## شروع سریع
```
python3.8 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install --upgrade --force-reinstall -r requirements.txt
```
اجرای Jupyter (روی سرور):
```
pip install jupyter ipykernel
python -m ipykernel install --user --name nprach_synch --display-name "Python 3.8 (nprach_synch)"
jupyter notebook --ip=0.0.0.0 --port 8888 --no-browser
```

## پیکربندی خودکار اجرا (جدید)
در ابتدای نوت‌بوک اضافه کنید:
```
from runtime.auto_config import get_system_profile, recommend_settings, apply_tf_settings, summarize
prof = get_system_profile(); rec = recommend_settings(prof, mode='eval'); apply_tf_settings(rec)
print(summarize(prof, rec))
```
این کار batch، نخ‌ها، mixed precision و … را معقول تنظیم می‌کند (XLA پیش‌فرض خاموش است).

## ساختار پروژه
- `nprach/`: موج NPRACH
- `synch/`: الگوریتم‌ها (Baseline, DeepNSynch)
- `e2e/`: مدل انتها‑به‑انتها (تولید، کانال، همگام‌سازی)
- `runtime/auto_config.py`: تنظیمات خودکار اجرا
- `parameters.py`: پارامترهای کلی (batch، بازه CFO و …)
- `Train.ipynb`, `Evaluate.ipynb`, `Evaluate_prepared.ipynb`
- `scripts/`: ابزارهای وزن/آزمون (generate/verify/train/smoke_test)

## نوت‌بوک‌ها (آموزش و ارزیابی)
- آموزش (Train.ipynb): `jit_compile=False` بماند مگر محیطتان سازگار باشد. بهتر است از auto_config برای انتخاب batch/threads استفاده کنید.
- ارزیابی (Evaluate*.ipynb):
  - Baseline بدون وزن اجرا می‌شود.
  - برای DeepNSynch باید `weights.dat` در ریشه باشد (وزن واقعی یا آزمایشی).
  - `Evaluate_prepared.ipynb` نسخه آماده برای تولید چهار شکل اصلی و ذخیره PNG/PDF در `results/` است.

## CPU/GPU و XLA
- هشدارهای CUDA/NUMA روی CPU بی‌اهمیت است.
- XLA پیش‌فرض خاموش است؛ فقط اگر کاملاً سازگار هستید روشن کنید.

## آزمون دود (Baseline)
سریع‌ترین تست سازگاری محیط:
```
python scripts/smoke_test.py
```

## تولید/بارگذاری وزن‌ها (DL – اختیاری)
ساخت وزن آزمایشی (برای اجرای گراف، نه نتایج مقاله):
```
python scripts/generate_weights.py
python scripts/verify_weights.py weights.dat   # اگر موجود بود
```
سپس در ارزیابی DL، قبل از استفاده، مدل را build کنید و وزن را set کنید.

## عیب‌یابی
- `ModuleNotFoundError: parameters`: Jupyter را از ریشه پروژه اجرا کنید (وارد کردن پارامترها در کد مقاوم‌سازی شده است).
- کندی/کمبود RAM روی CPU: `BATCH/RUNS` را کم کنید و برای Baseline از `NPRACH_NUM_SC=24` استفاده کنید.
- نسخه‌ها: طبق `requirements.txt` (TF 2.8.4 + Sionna 0.13.0). اگر GPU ندارید: `pip install tensorflow-cpu==2.8.4`.

## منابع
[A] https://arxiv.org/abs/2205.10805  — Deep Learning‑Based Synchronization for Uplink NB‑IoT

## مجوز
مطابق LICENSE.txt (LicenseRef‑NvidiaProprietary)

</div>
