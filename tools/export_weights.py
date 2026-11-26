# OpenAI CLIP ViT-B/32 の画像側を oshare 形式の .bin に書き出す（torch が要るのはここだけ）。
# 形式: b"OSHW" u32 版 u32 個数, 各テンソル: u16 名前長, 名前, u8 型(0=f32,1=int8行ごとscale), u8 次元数, u32[次元] 形, データ
#   型1: f32 scale[形[0]] のあと int8[要素数]  （2 次元の Linear 重みだけ）
import sys, struct, numpy as np, torch, open_clip
out = sys.argv[1]; quant = len(sys.argv) > 2 and sys.argv[2] == "int8"
m, _, _ = open_clip.create_model_and_transforms("ViT-B-32-quickgelu", pretrained="openai")
v = m.visual
sd = {k: t.detach().float().numpy() for k, t in v.state_dict().items()}
items = []
for k, a in sd.items():
    if quant and a.ndim == 2 and a.shape[0] >= 512 and a.shape[1] >= 512 and k != "positional_embedding" and k != "proj":
        s = np.abs(a).max(axis=1) / 127.0; s[s == 0] = 1
        q = np.clip(np.round(a / s[:, None]), -127, 127).astype(np.int8)
        items.append((k, 1, a.shape, s.astype(np.float32).tobytes() + q.tobytes()))
    else:
        items.append((k, 0, a.shape, a.astype(np.float32).tobytes()))
with open(out, "wb") as f:
    f.write(b"OSHW" + struct.pack("<II", 1, len(items)))
    for k, t, shp, data in items:
        kb = k.encode(); f.write(struct.pack("<H", len(kb)) + kb + struct.pack("<BB", t, len(shp)))
        f.write(struct.pack("<%dI" % len(shp), *shp)); f.write(data)
for k, t, shp, _ in items: print(k, "int8" if t else "f32", shp)
