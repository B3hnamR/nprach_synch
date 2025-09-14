<div dir="rtl" style="text-align: right">

# راهنمای راه‌اندازی و اجرای nprach_synch (فارسی و خودمونی)

## این پروژه چی کار می‌کند؟ (خیلی خلاصه)
- شبیه‌سازی شروع اتصال NB‑IoT را انجام می‌دهیم. دستگاه‌ها اول یک سیگنال به نام NPRACH می‌فرستند تا شبکه بفهمد «کی آمده؟» و دو چیز را تخمین بزند:
  - چه زمانی رسید؟ (ToA)
  - چقدر از فرکانس اصلی پرت شده؟ (CFO)
- دو مسیر همگام‌سازی داریم:
  - Baseline (تحلیلی/کلاسیک): بدون weight اجرا می‌شود و همین الان نتایج معتبر می‌دهد.
  - Deep Learning (DL/DeepNSynch): برای نتایج واقعی نیاز به weight آموزش‌داده دارد.

وضع فعلی: لینک weight مقاله دیگر در دسترس نیست. یک weight آزمایشی/سازگار ساخته‌ایم که فقط مدل DL اجرا شود؛ اما نمودارهای DL با این weight قابل استناد نیستند. برای تحویل رسمی، از مسیر Baseline استفاده کنید (نتایج قابل اتکاست). اگر لازم شد، بعداً DL را آموزش می‌دهیم و وزن واقعی تولید می‌کنیم.

---

## پیش‌نیازها
- Ubuntu 20.04/22.04 یا WSL2 روی ویندوز
- Python 3.8
- GPU اختیاری (CPU هم کفایت می‌کند، فقط کندتر است)

نصب Python 3.8 (پیشنهادی با deadsnakes):
```
apt update
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.8 python3.8-venv python3.8-dev
```

---

## شروع سریع (Quickstart)
```
cd /opt
git clone <REPO_URL> nprach_synch
cd nprach_synch

python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install --upgrade --force-reinstall -r requirements.txt
```

اجرای Jupyter روی سرور:
```
pip install jupyter ipykernel
python -m ipykernel install --user --name nprach_synch --display-name "Python 3.8 (nprach_synch)"
jupyter notebook --ip=0.0.0.0 --port 8888 --no-browser
# یا:
nohup jupyter notebook --ip=0.0.0.0 --port 8888 --no-browser > jupyter.log 2>&1 &
```

آزمون دود Baseline (سریع):
```
python scripts/smoke_test.py
```

---

## گرفتن خروجی‌های مقاله (نسخه سریع و قابل تحویل)
1) نوت‌بوک `Evaluate_prepared.ipynb` را باز کنید.
2) در سلول انتخاب روش:
   - `METHOD = 'baseline'`
   - برای CPU: `BATCH = 8..16` و `RUNS = 3..4`، و `NPRACH_NUM_SC = 24`
3) سلول‌های شکل‌ها را اجرا کنید. تصاویر ذخیره می‌شوند در:
   - `results/fpr_fnr_vs_cfo_baseline.(png|pdf)`
   - `results/fnr_vs_snr_baseline.(png|pdf)`
   - `results/fpr_fnr_vs_ptx_baseline.(png|pdf)`
   - `results/nmse_vs_snr_baseline.(png|pdf)`

نکته: هشدارهای CUDA/NUMA را نادیده بگیرید (GPU ندارید، مشکلی نیست).

---

## اگر می‌خواهید خط‌های DL هم روی نمودار باشد (اختیاری)
- چون weight مقاله نداریم، یا باید آموزش بدهیم یا فعلاً از weight آزمایشی استفاده کنیم.
- ساخت weight آزمایشی (فقط برای اجرای گراف، نه نتایج قابل استناد):
```
python scripts/generate_weights.py
```
- بعد در `Evaluate_prepared.ipynb` مقدار `METHOD = 'dl'` و سلول «بارگذاری وزن‌ها» را هم اجرا کنید. انتظار نداشته باشید نمودارها مثل مقاله شوند؛ برای نتایج جدی باید آموزش انجام شود.

---

## نکات عیب‌یابی و بهینه‌سازی
- ModuleNotFoundError: parameters
  - Jupyter را از ریشه پروژه (`/opt/nprach_synch`) اجرا کنید. در خود کد نیز import مقاوم شده است.
- کندی یا کمبود RAM روی CPU
  - `BATCH` و `RUNS` را پایین بیاورید و برای Baseline از `NPRACH_NUM_SC = 24` استفاده کنید.
- نسخه‌ها
  - از `tensorflow==2.8.4` و `sionna==0.13.0` طبق `requirements.txt` استفاده کنید.
  - اگر GPU ندارید: `pip install tensorflow-cpu==2.8.4`

---

## چیت‌شیت دستورات
```
# ساخت venv و نصب
python3.8 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# اجرای Jupyter
jupyter notebook --ip=0.0.0.0 --port 8888 --no-browser

# آزمون دود Baseline
python scripts/smoke_test.py

# ساخت weight آزمایشی (اختیاری برای DL)
python scripts/generate_weights.py
```

اگر اسکریپت/نوت‌بوک آماده‌ی بیشتری می‌خواهید، بفرمایید تا اضافه کنیم.

</div>

