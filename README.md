# oshare_cpp — おしゃれ度チェッカー（C++ / WebAssembly / Python）

服装の写真から「おしゃれ度（0〜100）」と近いスタイル（モード・ストリート・カジュアル…）を出す。
カメラで撮った写真でも、読み込んだ写真でもよい。推論はブラウザの中だけで行い、写真は外に送らない。

デモ: https://yomei-o.github.io/oshare_cpp/

## しくみ

- **OpenAI CLIP ViT-B/32**（MIT ライセンス）の画像側を、依存なしの C++ ヘッダ `cpp/oshare.hpp` で書き直した。
  Python 版 `python/oshare.py` は numpy だけで同じ計算をする（torch は要らない）。
- おしゃれ度 = 画像のベクトルと、「おしゃれな服装」の言葉 3 つ・「だらしない服装」の言葉 3 つとの
  類似度の差（`model/prompts_src.json`）。これをロジスティック関数で 0〜100 にする。
- スタイルタグ = 15 個のスタイルの言葉に対する softmax。
- 言葉側のベクトルは `tools/make_text.py`（torch の CLIP テキスト側）で前もって計算し、`model/prompts.json` に入れてある。
- 前処理は「長辺に合わせて平均色で正方形に余白を足す → 短辺 224 に PIL 互換の bicubic → 正規化」。
  bicubic は PIL と同じ 22bit 固定小数で、torch の前処理と**画素単位で一致**する。
- 重みは int8（出力ごとの scale）で 97MB。メモリ上でも int8 のまま持ち、計算のときに float に戻す。

最初は mmfashion を検討した。しかし mmfashion には「おしゃれ度」のモデルが無い。一番近い Fashion Compatibility は白背景の商品写真で学習されていて、着ている写真から服を切り出すのに Mask R-CNN も要る。なので CLIP にした。

## 数字（検証済み）

| 項目 | 結果 |
|---|---|
| numpy 版 vs open_clip（torch, f32） | 特徴の cos = 1.0000000、前処理の差 0 |
| int8 重み vs f32 | cos ≥ 0.9995 |
| C++ vs Python（`tools/parity.py`、同じ画素） | cos = 1.0000000、点数の差 ≤ 0.0001 |
| WASM（node, arm64 ノート PC） | 1 枚 1.1 秒、点数は C++ と一致 |
| ネイティブ x64（MSVC /arch:AVX2, 1 スレッド） | 1 枚 0.29 秒 |

モデルの比較（Commons の 136 枚：おしゃれ側＝渋谷スナップ・ストリートスタイル・ランウェイ 91 枚、普段着側＝スウェット・ジャージ 45 枚、ゼロショットの AUC）:

| モデル | AUC | 画像側 | ライセンス |
|---|---|---|---|
| MobileCLIP-S1 / S2 | 0.91 | 22M / 36M | Apple 研究専用 → 不採用 |
| **ViT-B/32 OpenAI (QuickGELU)** | **0.86**（余白方式 0.865） | 88M | MIT ← 採用 |
| ViT-B/16 LAION | 0.87 | 86M | MIT（計算は約 4 倍） |
| MobileCLIP2-S0 / S2 | 0.83〜0.84 | | Apple 研究専用 |
| ConvNeXt-B-W LAION | 0.79 | 88M | MIT |

**注意**：普段着側には古い白黒写真やスポーツの写真が多い。そのため AUC には「写真の新しさ・撮り方」の差も混ざっている。
136 枚では 0.03〜0.05 程度の差は誤差のうち。点数は「CLIP がおしゃれな服装の写真っぽいと思う度合い」であり、
人がつけたおしゃれ度で検証したものではない。

## 使い方

```sh
# C++（Windows は VS2022）
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build/Release/oshare_cli models/vitb32_int8.bin model/prompts.json photo.jpg

# Python（numpy + Pillow だけ）
python python/oshare.py models/vitb32_int8.bin model/prompts.json photo.jpg

# WASM（docs/oshare.js を作る）
./build_wasm.sh
```

重み `models/vitb32_int8.bin` は `docs/model/w0.bin` と `w1.bin` を連結したもの（`cat docs/model/w0.bin docs/model/w1.bin > models/vitb32_int8.bin`）。
作り直すときは `python tools/export_weights.py models/vitb32_int8.bin int8`（open_clip が要る）。

## ライセンス

- コード: MIT
- CLIP の重み: OpenAI CLIP（MIT）
- サンプル写真: Wikimedia Commons（CC BY 2.0 / パブリックドメイン、`docs/samples/credits.json` とページ下部にクレジット）
