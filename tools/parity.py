# C++ (oshare_cli) と Python (python/oshare.py) の突き合わせ。
# JPEG の復号は PIL と stb で違うので、PIL で復号した画素を PNG にしてから両方に渡す。
import sys, os, json, subprocess, tempfile, numpy as np
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import oshare
cli, wpath, ppath = sys.argv[1:4]; imgs = sys.argv[4:]
w = oshare.load_weights(wpath); pr = oshare.load_prompts(ppath)
tmp = tempfile.mkdtemp(); pngs = []
for i, fn in enumerate(imgs):
    p = os.path.join(tmp, "%03d.png" % i); Image.open(fn).convert("RGB").save(p); pngs.append(p)
out = subprocess.run([cli, wpath, ppath, *pngs, "--feat"], capture_output=True, text=True, check=True).stdout.split("\n")
worst_cos, worst_ds = 1.0, 0.0
for p, line in zip(pngs, out):
    fc = np.array([float(v) for v in line.split()], np.float32)
    rgb = np.asarray(Image.open(p).convert("RGB"))
    x = oshare.preprocess(rgb, mode=pr["preprocess"]); fp = oshare.encode_image(w, x)
    c = float(fc @ fp / np.linalg.norm(fc) / np.linalg.norm(fp))
    ds = abs(oshare.judge(pr, fc)["score"] - oshare.judge(pr, fp)["score"])
    worst_cos = min(worst_cos, c); worst_ds = max(worst_ds, ds)
    print("%s cos=%.8f  maxabs=%.2e  dscore=%.4f" % (os.path.basename(p), c, np.abs(fc - fp).max(), ds))
print("WORST cos=%.8f dscore=%.4f -> %s" % (worst_cos, worst_ds, "PASS" if worst_cos > 0.99999 and worst_ds < 0.01 else "FAIL"))
