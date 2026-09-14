# -*- coding: utf-8 -*-
"""
05_analyze.py
-------------
AC-CA - 5-fold sonuclarini toplar ve raporlanabilir istatistikleri uretir.

NEDEN AYRI SCRIPT: 04_train.py'nin sonundaki ozet yalnizca O OTURUMDA kosan
foldlari toplar. Devam ettirilmis kosularda atlanan foldlar ozette gorunmez.
Gercek sonuc her zaman buradan okunur.

Uretilenler:
  1. Fold tablosu (en iyi epoch, Dice, taban cizgisi, fark)
  2. 5-fold ortalama +/- standart sapma
  3. Alan-agirlikli Dice (kucuk lezyonlarin orantisiz etkisini dengeler)
  4. Eslesmis Wilcoxon testi: model vs prior-kopyala taban cizgisi
  5. Lezyon boyutuna gore stratifikasyon
  6. En cok kaybedilen ciftler (hata analizi icin)

Kullanim (Colab):
    import importlib.util, sys
    s = importlib.util.spec_from_file_location("A", "/content/acca/scripts/05_analyze.py")
    A = importlib.util.module_from_spec(s); sys.modules["A"] = A; s.loader.exec_module(A)
    A.analyze("eva02_prior")
    A.compare("eva02_prior", "eva02_noprior")     # ablasyon karsilastirmasi
"""

import os
import glob

import numpy as np
import pandas as pd

RUNS_DIR   = "/content/drive/MyDrive/acca_runs"
CACHE_ROOT = "/content/acca/_prior_cache"


def _cache_stats():
    """pair_stats.csv'yi bulur (lezyon alani icin gerekli)."""
    hits = glob.glob(os.path.join(CACHE_ROOT, "*", "pair_stats.csv"))
    if not hits:
        return None
    return pd.read_csv(hits[0], encoding="utf-8-sig")[["pair_id", "tgt_px", "cov_soft", "touch"]]


def load_run(run):
    """Bir kosunun fold ozetini ve cift bazli sonuclarini okur."""
    rd = os.path.join(RUNS_DIR, run)
    if not os.path.isdir(rd):
        raise FileNotFoundError(f"kosu klasoru yok: {rd}")

    m = pd.read_csv(os.path.join(rd, "metrics.csv"))
    # ayni fold birden fazla oturumda kosmus olabilir -> en iyi epoch'u al
    best = (m.sort_values("val_dice", ascending=False)
              .groupby("fold", as_index=False).first()
              .sort_values("fold"))

    pairs = []
    for f in sorted(glob.glob(os.path.join(rd, "fold*_val_pairs.csv"))):
        d = pd.read_csv(f, encoding="utf-8-sig")
        d["fold"] = int(os.path.basename(f).split("_")[0].replace("fold", ""))
        pairs.append(d)
    pairs = pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()

    st = _cache_stats()
    if st is not None and len(pairs):
        pairs = pairs.merge(st, on="pair_id", how="left")
    return best, pairs


def analyze(run="eva02_prior"):
    best, P = load_run(run)
    line = "=" * 68

    print(line); print(f"KOSU: {run}"); print(line)

    print("\n--- fold bazinda (en iyi epoch) ---")
    t = best[["fold", "epoch", "val_dice", "val_iou", "val_hd95",
              "baseline_dice", "delta"]].copy()
    t.columns = ["fold", "epoch", "Dice", "IoU", "HD95", "taban", "fark"]
    print(t.round(4).to_string(index=False))

    d, b = best.val_dice.values, best.baseline_dice.values
    print(f"\n  MODEL  Dice {d.mean():.4f} +/- {d.std(ddof=1):.4f}")
    print(f"  TABAN  Dice {b.mean():.4f} +/- {b.std(ddof=1):.4f}")
    print(f"  KAZANC      {d.mean()-b.mean():+.4f}   "
          f"(bagil {100*(d.mean()-b.mean())/b.mean():+.1f}%)")
    print(f"  IoU  {best.val_iou.mean():.4f}   HD95 {best.val_hd95.mean():.1f} px")

    if not len(P):
        print("\n[!] cift bazli dosya yok, ileri analiz atlandi.")
        return best, P

    print(f"\n--- cift bazli (n = {len(P)}) ---")
    diff = P.dice - P.dice_copy
    print(f"  goruntu-basi ortalama Dice : {P.dice.mean():.4f}")
    print(f"  medyan Dice                : {P.dice.median():.4f}")
    if "tgt_px" in P.columns and P.tgt_px.notna().any():
        w = P.tgt_px.fillna(0)
        print(f"  alan-agirlikli Dice        : {np.average(P.dice, weights=w):.4f}")
    print(f"  taban cizgisi ortalama     : {P.dice_copy.mean():.4f}")
    print(f"  ortalama fark              : {diff.mean():+.4f}")
    print(f"  medyan fark                : {diff.median():+.4f}")
    print(f"  modelin KAZANDIGI cift     : {(diff > 0).sum()}/{len(P)} "
          f"({100*(diff > 0).mean():.1f}%)")
    print(f"  modelin KAYBETTIGI cift    : {(diff < 0).sum()}/{len(P)}")

    try:
        from scipy.stats import wilcoxon
        s, p = wilcoxon(P.dice, P.dice_copy)
        # etki buyuklugu: eslesmis rank-biserial
        n = (diff != 0).sum()
        rb = 1 - 2 * s / (n * (n + 1) / 2)
        print(f"\n  Wilcoxon esleşmis siralar testi: W={s:.0f}, p={p:.3e}")
        print(f"  rank-biserial etki buyuklugu   : {rb:.3f}")
    except Exception as e:
        print(f"  [!] Wilcoxon hesaplanamadi: {e}")

    if "tgt_px" in P.columns and P.tgt_px.notna().any():
        print("\n--- lezyon boyutuna gore ---")
        bins = [0, 200, 800, 2000, 1e9]
        lab = ["<200 px", "200-800", "800-2000", ">2000"]
        P["boyut"] = pd.cut(P.tgt_px, bins=bins, labels=lab)
        g = P.groupby("boyut", observed=True).agg(
            n=("dice", "size"), model=("dice", "mean"),
            taban=("dice_copy", "mean"))
        g["fark"] = g.model - g.taban
        print(g.round(4).to_string())

    if "touch" in P.columns and P.touch.notna().any():
        print("\n--- prior hedefe degiyor mu ---")
        g = P.groupby("touch").agg(n=("dice", "size"), model=("dice", "mean"),
                                   taban=("dice_copy", "mean"))
        g["fark"] = g.model - g.taban
        print(g.round(4).to_string())

    print("\n--- modelin en cok kaybettigi 8 cift ---")
    cols = [c for c in ["pair_id", "fold", "dice", "dice_copy", "tgt_px", "hd95"]
            if c in P.columns]
    print(P.assign(fark=diff).nsmallest(8, "fark")[cols + ["fark"]]
           .round(3).to_string(index=False))
    print(line)
    return best, P


def compare(run_a, run_b):
    """Iki kosuyu cift bazinda eslesmis olarak karsilastirir (ablasyon)."""
    _, A = load_run(run_a)
    _, B = load_run(run_b)
    M = A[["pair_id", "dice"]].merge(B[["pair_id", "dice"]], on="pair_id",
                                     suffixes=("_a", "_b"))
    d = M.dice_a - M.dice_b
    print("=" * 68)
    print(f"KARSILASTIRMA  A={run_a}  vs  B={run_b}   (n={len(M)})")
    print("=" * 68)
    print(f"  A ortalama Dice : {M.dice_a.mean():.4f}")
    print(f"  B ortalama Dice : {M.dice_b.mean():.4f}")
    print(f"  fark (A - B)    : {d.mean():+.4f}   medyan {d.median():+.4f}")
    print(f"  A'nin kazandigi : {(d > 0).sum()}/{len(M)}")
    try:
        from scipy.stats import wilcoxon
        s, p = wilcoxon(M.dice_a, M.dice_b)
        print(f"  Wilcoxon        : W={s:.0f}, p={p:.3e}")
    except Exception as e:
        print(f"  [!] {e}")
    print("=" * 68)
    return M


if __name__ == "__main__":
    analyze("eva02_prior")
