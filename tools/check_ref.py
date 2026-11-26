# numpy 参照実装を open_clip（torch）と突き合わせる。前処理・特徴のずれと int8 の影響を見る。
import sys, json, os, numpy as np, torch, open_clip
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import oshare
meta = json.load(open(sys.argv[1], encoding="utf-8"))[::9]
m, _, prep = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai"); m.eval()
wf = oshare.load_weights(sys.argv[2]); wq = oshare.load_weights(sys.argv[3])
cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
for e in meta:
    im = Image.open(e["file"]).convert("RGB")
    xt = prep(im).numpy(); xn = oshare.preprocess(np.asarray(im))
    with torch.no_grad(): ft = m.encode_image(torch.from_numpy(xt)[None])[0].numpy()
    ff = oshare.encode_image(wf, xn); fq = oshare.encode_image(wq, xn)
    print("%-28s pre maxdiff=%.2e  f32 cos=%.7f  int8 cos=%.5f" % (e["file"][-24:], np.abs(xt - xn).max(), cos(ft, ff), cos(ft, fq)))
