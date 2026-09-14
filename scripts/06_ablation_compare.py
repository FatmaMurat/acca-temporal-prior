# -*- coding: utf-8 -*-
"""
06_ablation_compare.py
----------------------
AC-CA - prior ablasyonu: eslesmis karsilastirma ve alt grup analizi.

NEDEN ALT GRUP: ortalama fark (+0.04) tek basina yaniltici. Prior'in her
ciftte esit katki sagladigini varsayar. Oysa medyan fark ortalamadan cok
daha kucukse, kazanci birkac uc vaka tasiyor demektir. Asil savunulabilir
iddia "prior her yerde yardim ediyor" degil, "prior SU KOSULLARDA yardim
ediyor" seklindedir.

Kirilimlar:
  - lezyon boyutu (kucuklerde prior daha degerli olabilir)
  - prior hedefe degiyor mu (touch)
  - prior kapsamasi (cov_soft ceyrekleri)
  - taban cizgisi kalitesi (prior'in ne kadar bilgilendirici oldugu)

Her alt grup icin ayri Wilcoxon + Holm duzeltmesi (coklu karsilastirma).

Kullanim:
    import importlib.util, sys
    s = importlib.util.spec_from_file_location("C", "/content/acca/scripts/06_ablation_compare.py")
    C = importlib.util.module_from_spec(s); sys.modules["C"] = C; s.loader.exec_module(C)
    M = C.ablation("eva02_prior", "eva02_noprior")
"""

import os
import glob

import numpy as np
import pandas as pd

RUNS_DIR   = "/content/drive/MyDrive/acca_runs"
CACHE_ROOT = "/content/acca/_prior_cache"


def _load_pairs(run):
    rd = os.path.join(RUNS_DIR, run)
    fs = sorted(glob.glob(os.path.join(rd, "fold*_val_pairs.csv")))
    if not fs:
        raise FileNotFoundError(f"cift dosyasi yok: {rd}")
    out = []
    for f in fs:
        d = pd.read_csv(f, encoding="utf-8-sig")
        d["fold"] = int(os.path.basename(f).split("_")[0].replace("fold", ""))
        out.append(d)
    P = pd.concat(out, ignore_index=True)
    print(f"  {run:<18} {len(fs)} fold, {len(P)} cift")
    return P


def _cache_stats():
    hits = glob.glob(os.path.join(CACHE_ROOT, "*", "pair_stats.csv"))
    if not hits:
        print("  [!] pair_stats.csv bulunamadi -> alt grup analizi sinirli")
        return None
    return pd.read_csv(hits[0], encoding="utf-8-sig")[
        ["pair_id", "tgt_px", "cov_soft", "cov_bin", "touch", "aralik_gun"]]


def _wilcoxon(a, b):
    """(W, p, medyan_fark, kazanan_sayi) doner. Yetersiz n'de NaN."""
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    nz = d[d != 0]
    if len(nz) < 6:
        return np.nan, np.nan
    try:
        w, p = wilcoxon(a, b)
        return w, p
    except Exception:
        return np.nan, np.nan


def _holm(pvals):
    """Holm-Bonferroni duzeltmesi. NaN'lari korur."""
    p = np.asarray(pvals, dtype=float)
    ok = ~np.isnan(p)
    out = np.full_like(p, np.nan)
    idx = np.argsort(p[ok])
    vals = p[ok][idx]
    n = len(vals)
    adj = np.empty(n)
    run_max = 0.0
    for i, v in enumerate(vals):
        run_max = max(run_max, (n - i) * v)
        adj[i] = min(1.0, run_max)
    tmp = np.empty(n)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def _subgroup(M, col, labels, title):
    rows = []
    for lab in labels:
        g = M[M[col] == lab]
        if len(g) == 0:
            continue
        w, p = _wilcoxon(g.dice_a, g.dice_b)
        d = g.dice_a - g.dice_b
        rows.append(dict(grup=str(lab), n=len(g),
                         prior_var=g.dice_a.mean(), prior_yok=g.dice_b.mean(),
                         fark=d.mean(), medyan=d.median(),
                         kazanan=f"{(d > 0).sum()}/{len(g)}", p=p))
    if not rows:
        return None
    T = pd.DataFrame(rows)
    T["p_holm"] = _holm(T.p.values)
    print(f"\n--- {title} ---")
    disp = T.copy()
    for c in ("prior_var", "prior_yok", "fark", "medyan"):
        disp[c] = disp[c].map(lambda v: f"{v:.4f}")
    for c in ("p", "p_holm"):
        disp[c] = disp[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2e}")
    print(disp.to_string(index=False))
    return T


def ablation(run_a="eva02_prior", run_b="eva02_noprior", verbose=True):
    """A = prior'li, B = prior'siz. Eslesmis karsilastirma."""
    line = "=" * 74
    print(line); print(f"PRIOR ABLASYONU   A = {run_a}   B = {run_b}"); print(line)

    A, B = _load_pairs(run_a), _load_pairs(run_b)
    M = A[["pair_id", "fold", "dice", "dice_copy", "hd95"]].merge(
        B[["pair_id", "dice", "hd95"]], on="pair_id",
        suffixes=("_a", "_b"), validate="one_to_one")

    st = _cache_stats()
    if st is not None:
        M = M.merge(st, on="pair_id", how="left")

    if len(M) < len(A):
        print(f"  [!] {len(A)-len(M)} cift eslesmedi, karsilastirmadan cikarildi")

    d = M.dice_a - M.dice_b

    print(f"\n{'='*74}\nGENEL SONUC (n = {len(M)})\n{'='*74}")
    print(f"  prior VAR  ortalama Dice : {M.dice_a.mean():.4f}")
    print(f"  prior YOK  ortalama Dice : {M.dice_b.mean():.4f}")
    print(f"  taban cizgisi (kopyala)  : {M.dice_copy.mean():.4f}")
    print(f"\n  PRIOR'IN NET KATKISI     : {d.mean():+.4f}  "
          f"(medyan {d.median():+.4f})")
    print(f"  mimarinin katkisi        : {M.dice_b.mean()-M.dice_copy.mean():+.4f}"
          f"   (prior YOK vs taban)")
    tot = M.dice_a.mean() - M.dice_copy.mean()
    print(f"  toplam kazanc            : {tot:+.4f}")
    if tot > 0:
        print(f"     -> bunun %{100*d.mean()/tot:.0f}'i prior'dan, "
              f"%{100*(M.dice_b.mean()-M.dice_copy.mean())/tot:.0f}'i mimariden")

    w, p = _wilcoxon(M.dice_a, M.dice_b)
    print(f"\n  Wilcoxon (prior var vs yok): W={w:.0f}, p={p:.3e}")
    print(f"  prior'in kazandigi cift    : {(d > 0).sum()}/{len(M)} "
          f"({100*(d > 0).mean():.1f}%)")
    print(f"  prior'in kaybettirdigi     : {(d < 0).sum()}/{len(M)}")
    print(f"  |fark| > 0.10 olan cift    : {(d.abs() > 0.10).sum()}/{len(M)}")

    # HD95
    if {"hd95_a", "hd95_b"}.issubset(M.columns):
        h = M.dropna(subset=["hd95_a", "hd95_b"])
        if len(h):
            print(f"\n  HD95 prior VAR / YOK       : {h.hd95_a.mean():.1f} / "
                  f"{h.hd95_b.mean():.1f} px  (dusuk = iyi)")

    # ---------------- alt gruplar ----------------
    if "tgt_px" in M.columns and M.tgt_px.notna().any():
        M["boyut"] = pd.cut(M.tgt_px, [0, 200, 800, 2000, 1e9],
                            labels=["<200 px", "200-800", "800-2000", ">2000"])
        _subgroup(M, "boyut", ["<200 px", "200-800", "800-2000", ">2000"],
                  "LEZYON BOYUTUNA GORE")

    if "touch" in M.columns and M.touch.notna().any():
        M["degme"] = M.touch.map({True: "prior degiyor", False: "prior DEGMIYOR"})
        _subgroup(M, "degme", ["prior DEGMIYOR", "prior degiyor"],
                  "PRIOR HEDEFE DEGIYOR MU")

    if "cov_soft" in M.columns and M.cov_soft.notna().any():
        try:
            M["kapsama"] = pd.qcut(M.cov_soft, 4,
                                   labels=["Q1 (dusuk)", "Q2", "Q3", "Q4 (yuksek)"])
            _subgroup(M, "kapsama", ["Q1 (dusuk)", "Q2", "Q3", "Q4 (yuksek)"],
                      "PRIOR KAPSAMASINA GORE (cov_soft ceyrekleri)")
        except Exception:
            pass

    M["taban_kalite"] = pd.cut(M.dice_copy, [-0.01, 0.001, 0.3, 0.6, 1.01],
                               labels=["taban = 0", "taban 0-0.3",
                                       "taban 0.3-0.6", "taban > 0.6"])
    _subgroup(M, "taban_kalite",
              ["taban = 0", "taban 0-0.3", "taban 0.3-0.6", "taban > 0.6"],
              "TABAN CIZGISI KALITESINE GORE (prior ne kadar bilgilendirici)")

    print("\n--- prior'in EN COK yardim ettigi 8 cift ---")
    cols = [c for c in ["pair_id", "fold", "dice_a", "dice_b", "dice_copy",
                        "tgt_px", "touch"] if c in M.columns]
    print(M.assign(fark=d).nlargest(8, "fark")[cols + ["fark"]]
           .round(3).to_string(index=False))

    print("\n--- prior'in ZARAR verdigi 8 cift ---")
    print(M.assign(fark=d).nsmallest(8, "fark")[cols + ["fark"]]
           .round(3).to_string(index=False))

    print("\n" + line)
    print("YORUM REHBERI")
    print(line)
    print("""  - "fark" sutunu prior'in O ALT GRUPTAKI net katkisidir.
  - Ortalama fark ile medyan fark cok ayrisiyorsa kazanci birkac uc vaka
    tasiyor demektir; bu durumda alt grup tablosu ana sonuctan daha
    savunulabilirdir.
  - p_holm sutunu coklu karsilastirma icin duzeltilmis p degeridir;
    ham p degil, bunu raporlayin.
  - "taban = 0" ve "prior DEGMIYOR" satirlari en ilgi cekici olanlardir:
    prior'in yaniltici oldugu vakalarda model ne yapiyor?""")
    return M


def fold_table(run_a="eva02_prior", run_b="eva02_noprior"):
    """Fold bazinda yan yana tablo (makale Tablo 1 icin)."""
    rows = []
    for run, tag in ((run_a, "prior"), (run_b, "noprior")):
        m = pd.read_csv(os.path.join(RUNS_DIR, run, "metrics.csv"))
        b = (m.sort_values("val_dice", ascending=False)
               .groupby("fold", as_index=False).first().sort_values("fold"))
        for _, r in b.iterrows():
            rows.append(dict(fold=int(r.fold), kosu=tag, epoch=int(r.epoch),
                             dice=r.val_dice, iou=r.val_iou, hd95=r.val_hd95,
                             taban=r.baseline_dice))
    T = pd.DataFrame(rows)
    piv = T.pivot(index="fold", columns="kosu", values="dice")
    piv["taban"] = T[T.kosu == "prior"].set_index("fold").taban
    piv["prior_katkisi"] = piv["prior"] - piv["noprior"]
    print("\n--- FOLD BAZINDA (makale tablosu icin) ---")
    print(piv.round(4).to_string())
    print(f"\n  prior   : {piv['prior'].mean():.4f} +/- {piv['prior'].std(ddof=1):.4f}")
    print(f"  noprior : {piv['noprior'].mean():.4f} +/- {piv['noprior'].std(ddof=1):.4f}")
    print(f"  taban   : {piv['taban'].mean():.4f} +/- {piv['taban'].std(ddof=1):.4f}")
    print(f"  prior katkisi (fold ort.): {piv['prior_katkisi'].mean():+.4f} "
          f"+/- {piv['prior_katkisi'].std(ddof=1):.4f}")
    try:
        from scipy.stats import wilcoxon
        w, p = wilcoxon(piv["prior"], piv["noprior"])
        print(f"  fold duzeyinde Wilcoxon  : W={w:.0f}, p={p:.4f}  (n=5, dusuk guc)")
    except Exception:
        pass
    return piv


if __name__ == "__main__":
    fold_table()
    ablation()
