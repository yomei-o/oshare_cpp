# CLIP のゼロショットで「おしゃれ度」を出し、おしゃれ側／普段着側の AUC を測る。
import sys, json, torch, open_clip
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
meta = json.load(open(sys.argv[1], encoding="utf-8"))
POS = {"snap", "streetstyle", "catwalk"}
pos_p = ["a photo of a person in a stylish, fashionable outfit",
         "a photo of a well-dressed person with great fashion sense",
         "a street snap of a trendy, chic outfit"]
neg_p = ["a photo of a person in an unfashionable, sloppy outfit",
         "a photo of a poorly dressed person with no fashion sense",
         "a photo of a person in frumpy, plain everyday clothes"]
models = [("ViT-B-32-quickgelu", "openai")]
def auc(s, y):
    P = [a for a, b in zip(s, y) if b]; N = [a for a, b in zip(s, y) if not b]
    return sum((p > n) + 0.5 * (p == n) for p in P for n in N) / (len(P) * len(N))
for name, pre in models:
    try:
        m, _, prep = open_clip.create_model_and_transforms(name, pretrained=pre) if pre else open_clip.create_model_and_transforms(name)
        tok = open_clip.get_tokenizer(name)
    except Exception as e:
        print(name, pre, "load fail", e); continue
    m = m.cuda().eval()
    with torch.no_grad():
        t = m.encode_text(tok(pos_p + neg_p).cuda()); t = t / t.norm(dim=-1, keepdim=True)
        s, y = [], []
        for e in meta:
            x = prep(Image.open(e["file"]).convert("RGB")).unsqueeze(0).cuda()
            f = m.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
            sim = (100 * f @ t.T)[0]
            sc = (sim[:3].mean() - sim[3:].mean()).item()
            s.append(sc); y.append(e["label"] in POS)
    nparam = sum(p.numel() for p in m.visual.parameters()) / 1e6
    print(f"{name} {pre}: AUC={auc(s, y):.3f}  visual={nparam:.1f}M")
