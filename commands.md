# Commands Cheat Sheet (nprach_synch)

Environment (WSL2/Ubuntu)
- Create/activate venv
```
python3 -m venv .venv
source .venv/bin/activate
```
- Install deps
```
pip install --upgrade pip
pip install --upgrade --force-reinstall -r requirements.txt
```
- Register Jupyter kernel
```
python -m ipykernel install --user --name nprach_synch --display-name "Python 3.8 (nprach_synch)"
```
- Launch Jupyter
```
jupyter notebook --no-browser --port 8888
```
- Smoke test (baseline)
```
python - << 'PY'
from e2e import E2E
BATCH=16
sys = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=24, fft_size=256, pfa=0.999)
out = sys(BATCH, max_cfo_ppm=10., ue_prob=0.5)
print('OK:', len(out))
PY
```

WSL2 resource tuning (Windows)
- Create C:\\Users\\Behnam\\.wslconfig with:
```
[wsl2]
memory=20GB
swap=24GB
processors=8
```
- Apply and restart WSL:
```
wsl --shutdown
```

Git
```
# Pull latest
git pull --rebase origin main

# Save and push changes
git add -A
git commit -m "Update: env pins, baseline fixes, notebooks, docs"
git push origin main
```

Weights (DL)
```
# Option A: Place your own weights at project root as weights.dat
# Option B: Generate/verify shape‑compatible weights locally
python scripts/generate_weights.py
python scripts/verify_weights.py weights.dat
```

Notebook prologue (recommended)
```
from runtime.auto_config import get_system_profile, recommend_settings, apply_tf_settings, summarize
prof = get_system_profile()
rec  = recommend_settings(prof, mode='eval')
apply_tf_settings(rec)
print(summarize(prof, rec))
import tensorflow as tf
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(8)
```
