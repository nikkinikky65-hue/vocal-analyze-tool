# vocal-analyze-tool
歌声の倍音/HNRなどを解析するツール
# Voice Analyze (harmonics / band energy)

シンプルな歌声解析スクリプトです。  
1 本の `analyze.py` を実行するだけで、以下の指標をまとめて可視化します。

- 倍音比率 H1〜H5（平均値＋高HNRフレームでの Peak 分布）
- Band Energy（Low / Mid / High）
- HNR（Harmonic-to-Noise Ratio）
- Brightness（スペクトル重心）
- CPP（Cepstral Peak Prominence）
- H1A1 / H1A2
- SHR（Soft Harmonic Ratio）

出力は

- `analysis/summary_all.csv` … 全曲分のサマリ一覧
- `analysis/summary_plots/<曲名>_summary.png` … 1×2 の概要グラフ

のみです。  
時間推移グラフなどは削り、**「歌声のキャラクターをざっくり把握する」ことに特化**しています。

---

## 特徴

- 🎧 **音声ファイルを入れて実行するだけ**
  - `input/` フォルダに音源を置いて `python analyze.py` を叩くだけ
- 🔍 **倍音構造＋帯域バランスを一枚の画像で確認**
  - 左：H1〜H5 の平均比率と、高HNRフレームでの Peak 分布
  - 右：Low / Mid / High のエネルギー（dB）
- ⚠️ **2mix（歌＋伴奏）っぽい音源には注意書きを自動表示**
  - 「伴奏成分が強い2mixの可能性あり」と画像タイトルに表示
- 📄 **CSV 出力で後処理しやすい**
  - `summary_all.csv` に f0 / harmonics / band energy / HNR などを1行ずつ追記

---

## 動作環境

- Python 3.10 以降（開発時は 3.13 で動作確認）
- 推奨ライブラリ（例）:
  - `numpy`
  - `scipy`
  - `librosa` (0.10 系推奨)
  - `matplotlib`
  - `soundfile`
  - `audioread`

仮の `requirements.txt` 例：

```txt
numpy
scipy
librosa>=0.10,<1.0
matplotlib
soundfile
audioread
