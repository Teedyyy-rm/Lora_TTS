#!/usr/bin/env python3
"""Đánh giá giọng TTS 3 trục — thư viện chuyên nghiệp (Aug 2):
1. Phổ (analyze_audio logic — dải giọng nói %)  — audio SẠCH không
2. Speaker similarity (Resemblyzer)              — giọng GIỐNG mồi bao nhiêu %
3. WER/CER (faster-whisper)                      — phát âm ĐÚNG chữ không

Cách dùng:
  python3 evaluate_voice.py --wav out.wav --ref_wav vp_source.wav --text "câu gốc"
  python3 evaluate_voice.py --wav out.wav --ref_pkl vp.pkl --text "câu gốc"  (tự trích audio mồi? không — dùng --ref_wav)
"""
import argparse
import os
import sys

import numpy as np
import soundfile as sf


# ── 1. PHỔ (tái sử dụng logic analyze_audio.py) ──
def spectral_bands(wav_path):
    import scipy.fft as fft
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 24000:
        import torchaudio.functional as F
        import torch
        audio = F.resample(torch.from_numpy(audio).float(), sr, 24000).numpy()
        sr = 24000
    n = len(audio)
    if n < 1024:
        return 0.0
    window = np.hanning(n)
    spec = np.abs(fft.rfft(audio * window))
    freqs = fft.rfftfreq(n, 1 / sr)
    bands = [(0, 100), (100, 500), (500, 2000), (2000, 4000), (4000, 8000), (8000, 12000)]
    total = spec.sum() + 1e-10
    pcts = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        pcts.append(100.0 * spec[mask].sum() / total)
    speech = pcts[1] + pcts[2] + pcts[3]  # 100-4000Hz
    return speech


# ── 2. SPEAKER SIMILARITY (Resemblyzer / SpeechBrain ECAPA-TDNN) ──
_ecapa_classifier = None


def _get_ecapa():
    """Load ECAPA-TDNN (SpeechBrain, trained VoxCeleb — chuẩn benchmark).
    Nặng ~500MB — chỉ load khi --speaker_model ecapa."""
    global _ecapa_classifier
    if _ecapa_classifier is None:
        from speechbrain.inference.speaker import EncoderClassifier
        import os
        cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".speechbrain_cache")
        _ecapa_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=cache, run_opts={"device": "cuda" if _has_cuda() else "cpu"},
        )
    return _ecapa_classifier


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def speaker_similarity(wav_a, wav_b, model="resemblyzer"):
    """Cosine similarity 2 giọng. Resemblyzer: >0.75 = cùng giọng.
    ECAPA-TDNN: chuẩn khó hơn — cùng giọng thường >0.6, cần calibrate."""
    if model == "ecapa":
        import torch
        import torchaudio.functional as F
        clf = _get_ecapa()

        def embed(path):
            import soundfile as sf
            sig, sr = sf.read(path)
            if sig.ndim > 1:
                sig = np.mean(sig, axis=1)
            if sr != 16000:
                sig = F.resample(torch.from_numpy(sig).float(), sr, 16000).numpy()
            emb = clf.encode_batch(torch.tensor(sig).unsqueeze(0))
            return emb.squeeze().cpu().numpy()

        emb_a, emb_b = embed(wav_a), embed(wav_b)
    else:  # resemblyzer
        from resemblyzer import VoiceEncoder, preprocess_wav
        from pathlib import Path
        enc = VoiceEncoder()

        def embed(path):
            wav = preprocess_wav(Path(path))
            return enc.embed_utterance(wav)

        emb_a, emb_b = embed(wav_a), embed(wav_b)
    return float(np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b)))


# ── 3. WER/CER (faster-whisper) ──
def wer_score(wav_path, ref_text, model_size="large-v3"):
    from faster_whisper import WhisperModel
    import re
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, _ = model.transcribe(wav_path, language="vi")
    hyp = " ".join(s.text.strip() for s in segments)
    def norm(s):
        s = s.lower()
        s = re.sub(r"[^\w\sàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]", " ", s)
        return s.split()
    ref, h = norm(ref_text), norm(hyp)
    if not ref:
        return 1.0, hyp
    # Levenshtein
    dp = np.zeros((len(ref) + 1, len(h) + 1), dtype=np.int32)
    for i in range(len(ref) + 1): dp[i, 0] = i
    for j in range(len(h) + 1): dp[0, j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if ref[i-1] == h[j-1] else 1
            dp[i, j] = min(dp[i-1, j] + 1, dp[i, j-1] + 1, dp[i-1, j-1] + cost)
    wer = dp[len(ref), len(h)] / len(ref)
    return wer, hyp


def main():
    ap = argparse.ArgumentParser(description="Đánh giá TTS 3 trục: phổ + speaker sim + WER")
    ap.add_argument("--wav", required=True, help="WAV output TTS cần đánh giá")
    ap.add_argument("--ref_wav", required=True, help="WAV voice prompt gốc (đo giống giọng)")
    ap.add_argument("--text", required=True, help="Text gốc (đo WER)")
    ap.add_argument("--skip_wer", action="store_true", help="Bỏ qua WER (nhanh, không cần model)")
    ap.add_argument("--whisper_size", default="large-v3", help="Model whisper (large-v3/turbo/small)")
    ap.add_argument("--speaker_model", default="resemblyzer", choices=["resemblyzer", "ecapa"],
                    help="resemblyzer (nhẹ, nhanh, >0.75 = cùng giọng) | "
                         "ecapa (SpeechBrain VoxCeleb, chuẩn benchmark, nặng ~500MB, >0.6 = cùng giọng)")
    args = ap.parse_args()

    print(f"═══ ĐÁNH GIÁ: {os.path.basename(args.wav)} ═══")
    print(f"\n[1/3] Phổ (dải giọng nói 100-4000Hz)...")
    speech = spectral_bands(args.wav)
    verdict = "✅ SẠCH" if speech >= 70 else "⚠️ NHIỄU/MÉO" if speech < 60 else "🟡 TRUNG BÌNH"
    print(f"      Dải giọng: {speech:.0f}% → {verdict}")

    print(f"\n[2/3] Speaker similarity ({args.speaker_model})...")
    sim = speaker_similarity(args.wav, args.ref_wav, args.speaker_model)
    if args.speaker_model == "ecapa":
        sim_v = "✅ CÙNG GIỌNG" if sim > 0.6 else "🟡 GẦN GIỐNG" if sim > 0.45 else "❌ KHÁC GIỌNG"
    else:
        sim_v = "✅ CÙNG GIỌNG" if sim > 0.75 else "🟡 GẦN GIỐNG" if sim > 0.6 else "❌ KHÁC GIỌNG"
    print(f"      Cosine sim: {sim:.3f} → {sim_v}")

    if not args.skip_wer:
        print(f"\n[3/3] WER (faster-whisper {args.whisper_size})... (tải model lần đầu ~3GB)")
        wer, hyp = wer_score(args.wav, args.text, args.whisper_size)
        print(f"      WER: {wer*100:.1f}%")
        print(f"      STT: \"{hyp[:120]}\"")
    else:
        print(f"\n[3/3] Bỏ qua WER (--skip_wer)")

    print(f"\n════════ KẾT QUẢ ════════")
    print(f"  Dải giọng: {speech:.0f}% | Speaker sim: {sim:.3f}" + (f" | WER: {wer*100:.1f}%" if not args.skip_wer else ""))


if __name__ == "__main__":
    main()
