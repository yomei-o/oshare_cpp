# RESUME — 作業の記録と次の一手

## 2026-11-26 開始
- ユーザーの依頼: mmfashion を WASM C++ にして、カメラ・写真からおしゃれ度を判定する Web アプリ。
  そのあと「他のモデルやライブラリでもいい」と言われたので CLIP に切り替えた（理由は README）。
- 済んだこと: CLIP ViT-B/32 の画像側を C++ と numpy で実装。torch・C++・Python・WASM の 4 つで一致を確認。
  ブラウザ画面（写真を選ぶ・カメラ・サンプル）を作り、playwright でサンプル 5 枚が通ることを確認（`tools/browser_test.py`）。

## ビルド機（192.168.6.14）での配置
- `C:\prog\oshare` に同じ構成。`data\commons`（検証写真 136 枚、meta.json にライセンス）、`models\vitb32_{f32,int8}.bin`。
- open_clip_torch 4.2.2 を入れた。**`ViT-B-32` + `openai` は QuickGELU にならない**（警告だけ出て cos 0.97 になる）。
  必ず `ViT-B-32-quickgelu` を使う。

## 落とし穴
- Windows の ssh 先では `*.jpg` が展開されない。`for /r ... do` で回す。
- ssh 越しの python の標準出力は cp932。`sys.stdout.reconfigure(encoding="utf-8")` を入れる（-I では PYTHONIOENCODING が効かない）。
- Commons のサムネイル幅は 500 などの標準サイズにする（512 だと 429 が出やすい）。

## 次の一手（候補）
1. 人がつけたおしゃれ度のデータで目盛りを検証・学習する（今は CLIP のゼロショット）。
   写真の撮り方の偏りを避けるため、同じ撮影条件で服だけ違うデータが欲しい。
2. 人の検出で人物を切り出してから判定する（人が小さく写った写真に弱い）。
3. C++ に CLIP のテキスト側（＋BPE トークナイザ、tokenizers_cpp がある）を足し、
   言葉のベクトルを C++ でも作れるようにする（今は Python/torch だけ → パリティの穴）。
4. WASM のスレッド化（GitHub Pages は COOP/COEP を付けられないので coi-serviceworker が要る）。
5. iPhone 実機で未確認。
