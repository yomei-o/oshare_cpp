# prompts_src.json → 言葉のベクトル＋点数の目盛りを model/prompts.json に書く（torch が要る）。
# 目盛り: 画像ごとの差 s = mean(sim_pos) - mean(sim_neg) をロジスティックで 0..100 にする。
#   普段着側の中央値が 20 点、おしゃれ側の中央値が 80 点になるように mid と scale を決める。
import sys, json, os, numpy as np, torch, open_clip
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import oshare
src = json.load(open(sys.argv[1], encoding="utf-8")); meta = json.load(open(sys.argv[2], encoding="utf-8"))
w = oshare.load_weights(sys.argv[3]); out = sys.argv[4]
m, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai"); m.eval()
tok = open_clip.get_tokenizer("ViT-B-32-quickgelu")
def emb(ts):
    with torch.no_grad(): e = m.encode_text(tok(ts)).numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)
P = emb(src["score"]["pos"]); N = emb(src["score"]["neg"]); T = emb([t[1] for t in src["tags"]])
POS = {"snap", "streetstyle", "catwalk"}
def auc(s, y):
    Pp = [a for a, b in zip(s, y) if b]; Nn = [a for a, b in zip(s, y) if not b]
    return sum((p > n) + 0.5 * (p == n) for p in Pp for n in Nn) / (len(Pp) * len(Nn))
res = {}
for mode in ("crop", "pad"):
    S, Y = [], []
    for e in meta:
        rgb = np.asarray(Image.open(e["file"]).convert("RGB"))
        f = oshare.encode_image(w, oshare.preprocess(rgb, mode=mode)); f /= np.linalg.norm(f)
        S.append(float(100 * (P @ f).mean() - 100 * (N @ f).mean())); Y.append(e["label"] in POS)
    res[mode] = (S, Y); print(mode, "AUC=%.3f" % auc(S, Y))
mode = max(res, key=lambda k: auc(*res[k])); S, Y = res[mode]
sp = np.array([s for s, y in zip(S, Y) if y]); sn = np.array([s for s, y in zip(S, Y) if not y])
mid = float((np.median(sp) + np.median(sn)) / 2); scale = float((np.median(sp) - np.median(sn)) / (2 * np.log(4)))   # 中央値が 20 点と 80 点になる
print("mode", mode, "median pos %.3f neg %.3f -> mid %.3f scale %.3f" % (np.median(sp), np.median(sn), mid, scale))
r = lambda a: [round(float(x), 7) for x in a]
json.dump({"model": "openai CLIP ViT-B/32 (quickgelu)", "preprocess": mode, "logit_scale": 100.0,
           "score": {"pos": [r(x) for x in P], "neg": [r(x) for x in N], "mid": mid, "scale": scale},
           "tags": [{"label": t[0], "prompt": t[1], "emb": r(x)} for t, x in zip(src["tags"], T)]},
          open(out, "w", encoding="utf-8"), ensure_ascii=False)
