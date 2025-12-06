import numpy as np
import librosa
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import font_manager
import csv
import sys





plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False



BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"
ANALYSIS_DIR = BASE_DIR / "analysis"



# ============================
# A4-1: STFT
# ============================
def compute_stft(y, sr, n_fft=2048, hop_length=512):
    """
    旧版と同じ STFT（n_fft=2048, hop_length=512）
    """
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return S, freqs, hop_length





# ============================
# A4-3: extract_harmonics（精密版・時間変化対応）
# ============================
def extract_harmonics(S, freqs, f0):
    """
    1〜5倍音のエネルギー比率をフレームごとに算出。
    - 入力:
        S: 振幅スペクトル (freq_bins, frames)
        freqs: 周波数軸 [Hz]
        f0: 各フレームの基本周波数 [Hz]（pyin 結果）
    - 出力:
        H_mean: 各倍音(H1〜H5)の平均比率（全フレーム平均）
        H_all : shape=(5, frames) のフレームごとの比率
    """

    n_harm = 5
    n_frames = min(S.shape[1], len(f0))

    H_all = np.zeros((n_harm, n_frames), dtype=float)

    for t in range(n_frames):
        f0_t = f0[t]
        if np.isnan(f0_t) or f0_t <= 0:
            continue  # 無声フレームは 0 のまま

        # 各倍音の中心周波数（1〜5倍音）
        harmonics = [f0_t * n for n in range(1, n_harm + 1)]
        # ±10% の帯域
        bands = [(f * 0.9, f * 1.1) for f in harmonics]

        energy = []
        for low, high in bands:
            if high <= freqs[0] or low >= freqs[-1]:
                energy.append(0.0)
                continue
            mask = (freqs >= low) & (freqs <= high)
            band_energy = float(np.sum(S[mask, t]))
            energy.append(band_energy)

        energy = np.array(energy, dtype=float)
        total = np.sum(energy)
        if total > 0:
            H_all[:, t] = energy / total  # 比率に正規化

    # 平均比率（有効フレームのみ）
    valid_cols = np.where(H_all.sum(axis=0) > 0)[0]
    if len(valid_cols) == 0:
        H_mean = np.zeros(n_harm, dtype=float)
    else:
        H_mean = np.mean(H_all[:, valid_cols], axis=1)

    return H_mean, H_all


# ============================
# A5: Band Energy
# ============================
def compute_band_energy_db(S, freqs):
    """
    旧版 analyze.py と同じ band-energy の計算。
    - S: 振幅スペクトル
    - dBパワーへ変換して平均を取る
    """

    # まずパワーへ
    power = S ** 2

    # dBへ変換（旧版準拠の定義）
    power_db = 10 * np.log10(power + 1e-12)

    # 帯域
    low_mask = freqs < 500
    mid_mask = (freqs >= 500) & (freqs < 2000)
    high_mask = (freqs >= 2000) & (freqs < 8000)

    # 各帯域の平均(dB)
    low_energy = float(np.nanmean(power_db[low_mask]))
    mid_energy = float(np.nanmean(power_db[mid_mask]))
    high_energy = float(np.nanmean(power_db[high_mask]))

    return {
        "Low_dB": low_energy,
        "Mid_dB": mid_energy,
        "High_dB": high_energy
    }


def compute_band_energy(S, freqs):
    """
    旧コード互換のラッパー。
    内部では compute_band_energy_db を呼び、
    [Low, Mid, High] の3要素リストを返す。
    """
    d = compute_band_energy_db(S, freqs)
    return [d["Low_dB"], d["Mid_dB"], d["High_dB"]]


# ============================
# A6: HNR（自己相関ベース）
# ============================
def compute_hnr(y, sr, f0, frame_length=2048, hop_length=512, return_series=False):
    """
    自己相関ベース HNR（dB）
    - f0 が NaN でないフレームだけを有声として使用
    - return_series=True ならフレームごとの HNR 配列も返す
    """
    voiced_flag = ~np.isnan(f0)
    hnr_list = []
    hnr_series = np.full_like(f0, np.nan, dtype=float)

    for i, f0_i in enumerate(f0):
        if np.isnan(f0_i) or not voiced_flag[i]:
            continue

        center = i * hop_length
        start = center - frame_length // 2
        end = center + frame_length // 2

        if start < 0 or end > len(y):
            continue

        frame = y[start:end].astype(np.float64)
        frame = frame - np.mean(frame)
        if np.max(np.abs(frame)) < 1e-8:
            continue

        r = np.correlate(frame, frame, mode="full")
        mid = len(r) // 2
        r = r[mid:]
        R0 = r[0]
        if R0 <= 0:
            continue

        r = r / R0

        period = int(round(sr / f0_i))
        if period <= 0 or period >= len(r):
            continue

        R_tau = r[period]
        if R_tau <= 0 or R_tau >= 1:
            continue

        hnr_frame = 10.0 * np.log10(R_tau / (1.0 - R_tau + 1e-9))
        hnr_list.append(hnr_frame)
        hnr_series[i] = hnr_frame

    if not hnr_list:
        hnr_mean = float("nan")
    else:
        hnr_mean = float(np.mean(hnr_list))

    if return_series:
        return hnr_mean, hnr_series
    else:
        return hnr_mean


# ============================
# A6: Brightness
# ============================
def compute_brightness(S, sr, return_series=False):
    """
    Brightness（スペクトル重心）
    - mean: 旧版と同じ brightness
    - return_series=True でフレームごとの重心も返す
    """
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    centroid = np.nan_to_num(centroid)
    mean_c = float(np.mean(centroid))

    if return_series:
        return mean_c, centroid
    else:
        return mean_c

def compute_cpp(y, sr, f0, frame_length=2048, hop_length=512, return_series=False):
    """
    簡易版 CPP（Cepstral Peak Prominence）
    - y: 波形
    - sr: サンプリング周波数
    - f0: pyin で求めた基本周波数系列
    """
    f0 = np.asarray(f0)
    voiced = ~np.isnan(f0) & (f0 > 0)

    # 有声フレームが全くなければ NaN
    if not np.any(voiced):
        if return_series:
            return np.nan, np.full_like(f0, np.nan, dtype=float)
        else:
            return np.nan

    # フレーミング
    frames = librosa.util.frame(y, frame_length=frame_length,
                                hop_length=hop_length).T
    n_frames = min(len(frames), len(f0))

    cpp_vals = np.full(n_frames, np.nan, dtype=float)
    window = np.hamming(frame_length)

    for i in range(n_frames):
        if not voiced[i]:
            continue

        frame = frames[i] * window
        spec = np.fft.rfft(frame)
        log_power = np.log10(np.abs(spec) ** 2 + 1e-12)
        cepstrum = np.fft.irfft(log_power)

        f0_i = f0[i]
        if np.isnan(f0_i) or f0_i <= 0:
            continue

        # 基本周期 [サンプル] → ケプストラム上のインデックス
        period_samples = sr / f0_i
        q_center = int(round(period_samples))
        q_min = max(1, int(q_center * 0.8))
        q_max = min(len(cepstrum) // 2, int(q_center * 1.2))

        if q_max <= q_min + 1:
            continue

        peak = float(np.max(cepstrum[q_min:q_max]))
        # ベースラインはそれより手前の平均（超ざっくり版）
        baseline = float(np.mean(cepstrum[1:q_min])) if q_min > 1 else 0.0

        cpp_vals[i] = peak - baseline

    if np.any(~np.isnan(cpp_vals)):
        cpp_mean = float(np.nanmean(cpp_vals))
    else:
        cpp_mean = np.nan

    if return_series:
        series = np.full_like(f0, np.nan, dtype=float)
        series[:n_frames] = cpp_vals
        return cpp_mean, series

    return cpp_mean


def compute_h1_slope_features(H_all):
    """
    H_all: shape = (5, n_frames) の倍音比率 (0〜1)
    H1-A1, H1-A2 を簡易的に算出（H2, H3 を A1, A2 代わりとする）

    戻り値:
      H1A1_mean, H1A2_mean, H1A1_series, H1A2_series
    """
    H_all = np.asarray(H_all)
    n_harm, n_frames = H_all.shape

    # 倍音が3本未満なら計算できない
    if n_harm < 3:
        nan_series = np.full(n_frames, np.nan, dtype=float)
        return np.nan, np.nan, nan_series, nan_series

    H1 = H_all[0, :]
    H2 = H_all[1, :]
    H3 = H_all[2, :]

    eps = 1e-8
    H1_db = 20 * np.log10(H1 + eps)
    H2_db = 20 * np.log10(H2 + eps)
    H3_db = 20 * np.log10(H3 + eps)

    H1A1 = H1_db - H2_db
    H1A2 = H1_db - H3_db

    H1A1_mean = float(np.nanmean(H1A1))
    H1A2_mean = float(np.nanmean(H1A2))

    return H1A1_mean, H1A2_mean, H1A1, H1A2

def compute_shr(S, freqs, f0, return_series=False):
    """
    簡易版 SHR (Subharmonic-to-Harmonic Ratio)
    - S: 振幅スペクトル (freq_bins, frames)
    - freqs: 周波数軸 [Hz]
    - f0: 各フレームの基本周波数 [Hz]

    定義（簡易）:
      harmonics: k * f0 (k=1..4)
      subharm : (k+0.5) * f0 (k=0..3)  → 0.5f0, 1.5f0, 2.5f0, 3.5f0
      SHR_dB = 10 * log10( E_sub / (E_harm + eps) )
    """
    f0 = np.asarray(f0)
    n_bins, n_frames = S.shape
    n_frames = min(n_frames, len(f0))

    shr_vals = np.full(n_frames, np.nan, dtype=float)

    for t in range(n_frames):
        f0_t = f0[t]
        if np.isnan(f0_t) or f0_t <= 0:
            continue

        # 4つの倍音 & サブハーモニクスを見る
        harmonics = [f0_t * k for k in range(1, 5)]
        subharm   = [f0_t * (k + 0.5) for k in range(0, 4)]

        def band_energy(center_f):
            low = center_f * 0.9
            high = center_f * 1.1
            if high <= freqs[0] or low >= freqs[-1]:
                return 0.0
            mask = (freqs >= low) & (freqs <= high)
            return float(np.sum(S[mask, t]))

        E_h = sum(band_energy(f) for f in harmonics)
        E_s = sum(band_energy(f) for f in subharm)

        if E_h <= 0 or E_s <= 0:
            continue

        shr_vals[t] = 10.0 * np.log10(E_s / (E_h + 1e-12))

    if np.any(~np.isnan(shr_vals)):
        shr_mean = float(np.nanmean(shr_vals))
    else:
        shr_mean = np.nan

    if return_series:
        series = np.full_like(f0, np.nan, dtype=float)
        series[:n_frames] = shr_vals
        return shr_mean, series

    return shr_mean





# ============================
# A6: summary.csv & summary_all.csv
# ============================



def append_global_summary(song_name, H_mean, bands, hnr, bright,
                          cpp, h1a1, h1a2, shr,
                          f0_min, f0_max, f0_mean, f0_std,
                          global_path=ANALYSIS_DIR / "summary_all.csv"):



    """
    全曲分のまとめCSV（summary_all.csv）に1行追記する。
    ファイルがなければヘッダー付きで新規作成。
    """
    global_path = Path(global_path)
    global_path.parent.mkdir(exist_ok=True)

    file_exists = global_path.exists()

    with open(global_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        # 初回だけヘッダーを書く
        if not file_exists:
            writer.writerow([
                "song",
                "f0_min", "f0_max", "f0_mean", "f0_std",
                "H1", "H2", "H3", "H4", "H5",
                "Low", "Mid", "High",
                "HNR", "Brightness",
                "CPP", "H1_A1", "H1_A2",
                "SHR",
            ])



        writer.writerow([
            song_name,
            float(f0_min), float(f0_max), float(f0_mean), float(f0_std),
            float(H_mean[0]), float(H_mean[1]), float(H_mean[2]),
            float(H_mean[3]), float(H_mean[4]),
            float(bands[0]), float(bands[1]), float(bands[2]),
            float(hnr), float(bright),
            float(cpp), float(h1a1), float(h1a2),
            float(shr),
        ])


# ============================
# A6: Plot
# ============================
def plot_summary_merged(summary_root, song_label,
                        H_mean, bands, hnr, bright, cpp, shr,
                        H_all, hnr_series):
    """1×2 のまとめ画像を作成して保存する。

    [左] Harmonics mean (H1〜H5) ＋
         Peak harmonic energy distribution（高HNRフレーム）
    [右] Band Energy (Low/Mid/High)

    保存先: summary_root / f"{song_label}_summary.png"
    """
    summary_root = Path(summary_root)
    summary_root.mkdir(parents=True, exist_ok=True)

    # 長さを揃える（Peak 計算にだけ使う）
    n = min(H_all.shape[1], len(hnr_series))
    H = H_all[:, :n]          # shape = (5, n)
    h = hnr_series[:n]        # shape = (n,)

    # --- Peak 時の harmonic energy distribution を算出 ---
    voiced = ~np.isnan(h)
    if np.any(voiced):
        h_voiced = h[voiced]
        try:
            thr = np.percentile(h_voiced, 75)
        except Exception:
            thr = np.nanmax(h_voiced)

        peak_mask = voiced & (h >= thr)

        if peak_mask.sum() >= 5:
            H_peak = np.nanmean(H[:, peak_mask], axis=1)
        else:
            H_peak = np.nanmean(H[:, voiced], axis=1)
    else:
        H_peak = np.array(H_mean, copy=True)

    # ★ ここから 1×2 レイアウト
    fig, (ax00, ax01) = plt.subplots(1, 2, figsize=(12, 4))

    labels_h = ["H1", "H2", "H3", "H4", "H5"]

    # === [左] 平均ハーモニクス ＋ Peak分布 ===
    mean_pct = np.array(H_mean) * 100.0
    peak_pct = np.array(H_peak) * 100.0

    bars = ax00.bar(labels_h, mean_pct, color="skyblue", alpha=0.7, label="Mean")
    x_pos = [b.get_x() + b.get_width() / 2 for b in bars]

    ax00.plot(
        x_pos, peak_pct,
        marker="o", linestyle="-", color="orange",
        label="Peak (high HNR)"
    )

    if np.isfinite(mean_pct.max()) and np.isfinite(peak_pct.max()):
        ymax = float(max(mean_pct.max(), peak_pct.max()))
    else:
        ymax = 100.0
    ax00.set_ylim(0, ymax * 1.15)

    # ★ 2mix ワーニング判定
    low, mid, high = bands
    low_mid = 0.5 * (low + mid)
    diff = low_mid - high
    is_mixture_like = (
        (diff > 8.0 and hnr < 8.0) or
        (diff > 5.0 and hnr < 0.0)
    )

    title = (
        f"{song_label}\n"
        f"HNR={hnr:.1f} dB / Brightness={bright:.0f} Hz / "
        f"SHR={shr:.2f} dB / CPP={cpp:.3f}"
    )
    if is_mixture_like:
        title += "\n※伴奏成分が強い2mixの可能性あり"

    ax00.set_ylabel("Ratio (%)")
    ax00.set_title(title)

    # Mean のパーセントラベル
    for x, m_val, p_val in zip(x_pos, mean_pct, peak_pct):
        ax00.text(
            x, m_val - 2,
            f"{m_val:.1f}%",
            ha="center", va="top",
            fontsize=8, color="black"
        )

    ax00.legend(loc="upper right")

    # === [右] Band Energy (Low/Mid/High) ===
    band_labels = ["Low", "Mid", "High"]
    bars_be = ax01.bar(band_labels, bands)

    ax01.set_ylabel("Energy (dB)")
    ax01.set_title("Band Energy")

    for bar, val in zip(bars_be, bands):
        x = bar.get_x() + bar.get_width() / 2
        if val < 0:
            text_y = val - 0.5
            va = "top"
        else:
            text_y = val + 0.5
            va = "bottom"

        ax01.text(
            x, text_y,
            f"{val:.1f} dB",
            ha="center", va=va,
            fontsize=8,
        )

    fig.tight_layout()
    out_path = summary_root / f"{song_label}_summary.png"
    fig.savefig(out_path)
    plt.close(fig)




def load_audio(path: Path):
    # 録音形式は問わず librosa に丸投げ
    y, sr = librosa.load(path, sr=None, mono=True)
    return y, sr




# ============================
# メイン: 1曲の wav を解析
# ============================
def analyze_wav(wav_path: Path):
    wav_path = Path(wav_path)
    song_name = wav_path.stem


    # ★ 曲ごとのサブフォルダはやめて、analysis/ 直下だけ確保
    out_dir = ANALYSIS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)


    # A3: wav ロード
    y, sr = load_audio(wav_path)

    # A3: f0（pyin）
    fmin, fmax = 80, 1000
    f0, voiced_flag, voiced_probs = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr
    )
    # f0 が全部 NaN ならそのまま NaN のまま処理（HNRや倍音は NaN/0 になる）
    if np.isnan(np.nanmean(f0)):
        print(f"⚠ f0 が取得できませんでした: {wav_path.name}")

    # ★ f0 の統計量（min / max / mean / std）を算出しておく
    f0_arr = np.asarray(f0, dtype=float)
    f0_valid = f0_arr[(f0_arr > 0) & np.isfinite(f0_arr)]
    if f0_valid.size > 0:
        f0_min  = float(f0_valid.min())
        f0_max  = float(f0_valid.max())
        f0_mean = float(f0_valid.mean())
        f0_std  = float(f0_valid.std())
    else:
        f0_min = f0_max = f0_mean = f0_std = float("nan")


    # A4: STFT
    S, freqs, hop = compute_stft(y, sr)

    # A4: 倍音
    H_mean, H_all = extract_harmonics(S, freqs, f0)

    # A5: Band Energy
    bands = compute_band_energy(S, freqs)

    # A6: HNR（平均＋系列）
    hnr, hnr_series = compute_hnr(
        y, sr, f0,
        frame_length=2048,
        hop_length=hop,
        return_series=True
    )

    # A6: Brightness（平均＋系列）
    bright, centroid_series = compute_brightness(S, sr, return_series=True)

    # ★ CPP（平均＋系列）
    cpp, cpp_series = compute_cpp(
        y, sr, f0,
        frame_length=2048,
        hop_length=hop,
        return_series=True
    )

    # ★ H1-A1 / H1-A2
    h1a1_mean, h1a2_mean, h1a1_series, h1a2_series = compute_h1_slope_features(H_all)


    # ★ SHR（平均＋系列）
    shr, shr_series = compute_shr(S, freqs, f0, return_series=True)


    
    # 全体まとめCSVに1行追加
    append_global_summary(
        song_name,
        H_mean, bands, hnr, bright,
        cpp, h1a1_mean, h1a2_mean, shr,
        f0_min, f0_max, f0_mean, f0_std,
    )



    # ★ まとめ画像（harmonics + time_series）を共通フォルダに出力
    summary_root = ANALYSIS_DIR / "summary_plots"
    plot_summary_merged(
        summary_root,
        song_name,
        H_mean, bands, hnr, bright, cpp, shr,
        H_all, hnr_series,
    )


    print(f"✓ 解析完了：{wav_path.name}")





# ============================
# 実行エントリポイント
# ============================
if __name__ == "__main__":
    # 解析対象のベースフォルダ
    #   引数あり → input/配下のそのフォルダ
    #   引数なし → input/ 直下
    if len(sys.argv) > 1:
        base_dir = INPUT_DIR / sys.argv[1]
    else:
        base_dir = INPUT_DIR

    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"解析対象フォルダが見つかりません: {base_dir}")

    # 対象ファイル一覧（直下のみ）
    SUPPORTED_EXT = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".ogg"}
    audio_files = sorted(
        f for f in base_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT
    )

    if not audio_files:
        raise FileNotFoundError(f"{base_dir} 直下に対応する音声ファイルがありません")

    # 実行のたびに summary_all.csv をリセット
    global_summary = ANALYSIS_DIR / "summary_all.csv"
    if global_summary.exists():
        global_summary.unlink()

    print("=== analyze START ===")
    print(f"解析対象フォルダ: {base_dir}")
    print(f"音声ファイル数: {len(audio_files)}")

    for audio_path in audio_files:
        analyze_wav(audio_path)

    print("=== analyze COMPLETED ===")

