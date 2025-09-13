# 1) ساخت وزن‌ها بدون آموزش (فقط اینیت) — سه خروجی می‌دهد
python scripts/generate_weights.py
# => weights.dat , weights.npz , weights.h5

# 2) راستی‌آزمایی هرکدوم از فرمت‌ها
python scripts/verify_weights.py weights.dat
python scripts/verify_weights.py weights.npz
python scripts/verify_weights.py weights.h5

# 3) (اختیاری) مسیر آموزش شما
python scripts/train_deepnsynch.py   # فعلاً warm-build؛ می‌تونی حلقه‌ی آموزش واقعی رو اضافه کنی




✅ تغییرات اعمال‌شده

generate_weights.py – کاملاً بازنویسی شده:

وزن‌ها را در سه فرمت مختلف ذخیره می‌کند:

weights.dat (pickle list قدیمی برای سازگاری)

weights.npz (نگاشت نام→آرایه برای بارگذاری ایمن)

weights.h5 (فرمت استاندارد Keras)

برای Sionna یک ماژول ساده ساخته می‌شود تا وابستگی به نصب آن برداشته شود.

یک متادیتا (weights_meta.json) نیز ذخیره می‌کند که شامل تعداد وزن‌ها و طول ورودی است.

همچنین حاوی توضیح کامل نحوه کار و API است.

train_deepnsynch.py – به‌روزرسانی کامل:

علاوه بر آموزش یا Warm-build، وزن‌های DeepNSynch را بعد از هر چند گام به سه فرمت ذخیره می‌کند.

فایل‌های دوره‌ای با نام weights_step<step>.dat در پوشه results/ نگه‌داری می‌شوند.

اگر Sionna نصب نباشد، همچنان اجرا می‌شود.

اسکریپت‌های جدید

load_weights.py: یک لودر چندفرمتی که می‌تواند وزن‌ها را از .npz، .h5 یا .dat بخواند و در مدل ست کند.

verify_weights.py: یک ابزار سریع برای ساخت مدل dummy، بارگذاری وزن‌ها (از هر فرمتی) و اجرای یک forward pass برای اطمینان از سازگاری.


⏭️ پیشنهاد استفاده

ابتدا با اجرای generate_weights.py در محیط پروژه، سه فایل وزن تولید کنید.

با verify_weights.py <weights.dat|weights.npz|weights.h5> تست کنید که فایل‌ها بدون خطا بارگذاری می‌شوند.

در صورت نیاز به آموزش واقعی، از train_deepnsynch.py استفاده کنید تا پس از هر دوره، وزن‌ها به‌روز شوند.