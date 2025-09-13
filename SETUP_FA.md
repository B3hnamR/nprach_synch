# راهنمای راه‌اندازی و اجرای پروژه nprach_synch (فارسی)

این راهنما تمام مراحل لازم برای آماده‌سازی، اجرا و تولید خروجی‌های اصلی پروژه را روی لینوکس (Ubuntu/WSL2) توضیح می‌دهد. هدف اجرای سریع مسیر مبنا (Baseline) و در صورت نیاز آماده‌سازی مسیر DL است.

---

## پیش‌نیازهای سیستم
- Ubuntu 20.04 یا 22.04 (یا WSL2 روی ویندوز)
- Python 3.8
- GPU اختیاری (برای سرعت آموزش DL). روی CPU هم قابل اجراست.

### نصب Python 3.8 (Ubuntu)
روش پیشنهادی (PPA deadsnakes):
```
apt update
apt install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt update
apt install -y python3.8 python3.8-venv python3.8-dev
python3.8 --version
```
روش جایگزین (Snap):
```
snap install python38
# سپس به جای python3.8 از python38 استفاده کنید
```

---

## کلون پروژه و ساخت محیط مجازی
```
cd /opt
git clone <REPO_URL> nprach_synch
cd nprach_synch

python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## نصب وابستگی‌ها
وابستگی‌ها در `requirements.txt` پین شده‌اند (TF 2.8.4 + Sionna 0.13.0). نصب کامل:
```
pip install --upgrade --force-reinstall -r requirements.txt
```
اگر فقط می‌خواهید وزن‌ها را بسازید/تأیید کنید و اجرای کامل لازم نیست (گزینه سبک):
```
pip install tensorflow-cpu==2.8.4 numpy==1.22.4
```

---

## راه‌اندازی Jupyter روی سرور
نصب Jupyter و ثبت کرنل (داخل venv):
```
pip install jupyter ipykernel
python -m ipykernel install --user --name nprach_synch --display-name "Python 3.8 (nprach_synch)"
```
اجرای Jupyter (با دسترسی شبکه):
```
jupyter notebook --ip=0.0.0.0 --port 8888 --no-browser
# یا در پس‌زمینه:
nohup jupyter notebook --ip=0.0.0.0 --port 8888 --no-browser > jupyter.log 2>&1 &
```
لینک با token را در مرورگر باز کنید: `http://SERVER_IP:8888/?token=...`

---

## تنظیمات پیشنهادی در ابتدای نوت‌بوک
در ابتدای نوت‌بوک‌ها جهت تنظیم نخ‌ها و چاپ پروفایل سیستم:
```python
from runtime.auto_config import get_system_profile, recommend_settings, apply_tf_settings, summarize
prof = get_system_profile(); rec = recommend_settings(prof, mode='eval'); apply_tf_settings(rec)
print(summarize(prof, rec))
```

---

## آزمون دود (Smoke Test) مسیر مبنا
سریع‌ترین تست سازگاری محیط بدون وزن‌ها:
```
python scripts/smoke_test.py
```
یا در یک سلول/اسکریپت پایتون:
```python
from e2e import E2E
sys = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=24, fft_size=256, pfa=0.999)
out = sys(16, max_cfo_ppm=10.0, ue_prob=0.5)
print('OK:', len(out))
```

---

## تولید و تأیید وزن‌ها (اختیاری برای DL)
اگر وزن‌های مقاله را ندارید، می‌توانید وزن‌های سازگار با شکل را محلی بسازید:
```
python scripts/generate_weights.py
# خروجی: weights.dat / weights.npz / weights.h5 / weights_meta.json
```
تأیید (اگر اسکریپت موجود است):
```
python scripts/verify_weights.py weights.dat
```

توجه: این وزن‌ها آموزشی نیستند و برای نتایج معتبر DL کافی نیستند؛ فقط جهت اجرای گراف و سازگاری شکل‌ها کاربرد دارند. برای نتایج مقاله باید آموزش دهید یا وزن‌های آموزش‌داده را داشته باشید.

---

## تولید شکل‌ها و خروجی‌ها (Baseline/DL)
نوت‌بوک آماده `Evaluate_prepared.ipynb` را باز کنید و سلول‌ها را به‌ترتیب اجرا کنید.
- برای خروجی سریع و معتبر روی CPU:
  - در سلول انتخاب روش: `METHOD = 'baseline'`
  - `BATCH = 8..16`، `RUNS = 3..4`، و برای baseline مقدار `NPRACH_NUM_SC = 24`
- خروجی‌ها به‌صورت خودکار ذخیره می‌شوند:
  - `results/fpr_fnr_vs_cfo_baseline.(png|pdf)`
  - `results/fnr_vs_snr_baseline.(png|pdf)`
  - `results/fpr_fnr_vs_ptx_baseline.(png|pdf)`
  - `results/nmse_vs_snr_baseline.(png|pdf)`

برای اجرای DL (اختیاری):
1) فایل `weights.dat` را در ریشه بگذارید (یا بسازید).
2) در سلول بارگذاری وزن‌ها، مدل یک بار با ورودی ساختگی build می‌شود و سپس وزن‌ها set می‌شوند.
3) `METHOD = 'dl'` را انتخاب کنید و همان شکل‌ها را اجرا کنید. (کیفیت خروجی وابسته به کیفیت وزن‌هاست.)

---

## آموزش ساده‌ی DL (اختیاری)
نوت‌بوک `Train.ipynb` را باز کنید یا یک حلقه‌ی ساده آموزش اضافه کنید (نمونه‌ی آماده در گفت‌وگو آمده است). برای نتایج نزدیک مقاله به آموزش طولانی‌تر و GPU نیاز است.

---

## نکات عیب‌یابی
- هشدارهای CUDA/NUMA: روی CPU قابل صرف‌نظر است.
- `ModuleNotFoundError: parameters`: از ریشه پروژه Jupyter را اجرا کنید (`/opt/nprach_synch`) یا PYTHONPATH را تنظیم کنید:
  - `export PYTHONPATH=/opt/nprach_synch:$PYTHONPATH`
  (در کد پروژه، import پارامترها مقاوم‌سازی شده است.)
- حافظه/سرعت CPU: `BATCH` و `RUNS` را کاهش دهید و برای baseline از `NPRACH_NUM_SC=24` استفاده کنید.
- نسخه‌ها: از `tensorflow==2.8.4` و `sionna==0.13.0` طبق `requirements.txt` استفاده کنید. اگر TF GPU ندارید، از نسخه CPU استفاده کنید:
  - `pip install tensorflow-cpu==2.8.4`

---

## خلاصه دستورات متداول
```
# ساخت venv و نصب
python3.8 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# اجرای Jupyter
jupyter notebook --ip=0.0.0.0 --port 8888 --no-browser

# آزمون دود baseline
python scripts/smoke_test.py

# تولید/تأیید وزن‌ها (اختیاری)
python scripts/generate_weights.py
python scripts/verify_weights.py weights.dat
```

موفق باشید! اگر نیاز به اسکریپت‌های آماده‌ی بیشتری برای ارزیابی یا آموزش دارید، بفرمایید تا اضافه شوند.

