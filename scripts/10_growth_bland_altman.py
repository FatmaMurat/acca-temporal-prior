# -*- coding: utf-8 -*-
"""
10_growth_bland_altman.py
-------------------------
AC-CA - Adim 10: buyume niceliklendirmesi + Bland-Altman.

NE OLCULUYOR
  Model yalnizca t anini tahmin eder; t-1 maskesi zaten elimizdedir (prior'in
  kaynagi). Bu yuzden MUTLAK buyume farki uzerinde Bland-Altman yapmak
  anlamsizdir:

      tahmin buyume - referans buyume
        = [A(pred_t) - A(GT_{t-1})] - [A(GT_t) - A(GT_{t-1})]
        = A(pred_t) - A(GT_t)

  yani t anindaki alan hatasiyla BIREBIR ayni grafik cikar. Bu betik bu yuzden
  iki gercek uc nokta uretir:

    (1) YUZDE DEGISIM  %d = 100 * (A_t - A_{t-1}) / A_{t-1}
        t-1'e bolme dogrusal degildir; hatayi lezyon boyutuna gore yeniden
        olcekler (RECIST mantigi). Bland-Altman burada yapilir.
    (2) KATEGORIK SINIFLAMA  buyudu / stabil / kuculdu + Cohen kappa
        Radyolog da esikli bir karar verir; klinik olarak en savunulabilir
        cikti budur.

  Mutlak alan Bland-Altman'i EK olarak uretilir (= alan hatasi), ana sonuc degil.

BIRIM
  Tum alanlar mm^2. DFOV hastalar arasi degisir (55 ciftte olcek orani > 1.20),
  px^2 alanlari hastalar arasinda toplanamaz.
    A_prev  : prev_mask NATIF cozunurlukte okunur, prev_mmpx^2 ile carpilir.
    A_curr  : 512 izgarasinda (modelin gordugu geometri), mm_per_px_512^2 ile.
  GT_t ve pred_t ayni izgarada oldugu icin aralarinda yeniden orneklem hatasi
  yoktur; A_prev farkli bir goruntudur, natif okunmasi daha dogrudur.

  2B TEK KESIT: DICOM yok, kesit kalinligi bilinmiyor -> HACIM DEGIL ALAN.
  "Buyume" burada eslesmis kesitte 2B alan degisimidir. Sinirlilik olarak yazilir.

VARSAYIM KONTROLLERI (hakem bunlari sorar)
  - Orantisal yanlilik: fark ~ ortalama regresyonu. Anlamli ise klasik LoA
    gecersizdir; log donusumu / regresyon tabanli LoA gerekir.
  - Normallik: farklarin Shapiro-Wilk testi. Ihlal varsa yuzdelik tabanli LoA.
  - Tekrarli olcum: 244 cift 136 hastadan gelir, bagimsiz degildir.
    Hasta basina tek cift ile duyarlilik analizi yapilir.

Kullanim (Colab):
    V = load("G", "/content/acca/scripts/10_growth_bland_altman.py")
    df = V.run(model="sam2_prior", sam2_ckpt="/content/sam2.1_hiera_base_plus.pt")
"""

import os
import sys
import importlib.util

import numpy as np
import pandas as pd
import cv2
import torch

BASE_DIR   = "/content/acca"
CACHE_ROOT = "/content/acca/_prior_cache"
SCRIPTS    = "/content/acca/scripts"
RUNS_DIR   = "/content/drive/MyDrive/acca_runs"
OUT_DIR    = "/content/drive/MyDrive/acca_runs/_growth"

# kategorik esik: alan yuzde degisimi
# RECIST 1.1 CAP icin +%20 / -%30 kullanir; alanda yaklasik karsiligi
# (1.20^2-1)=+%44 ve (0.70^2-1)=-%51. Varsayilan olarak bu alinir,
# ayrica esik taramasi raporlanir.
GROW_TH   = +44.0
SHRINK_TH = -51.0


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


def imread_u(path):
    """Turkce karakterli yollarda cv2.imread sessizce basarisiz olur."""
    a = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(a, cv2.IMREAD_GRAYSCALE)


# ====================== 1) cikarim: tahmin alanlari ======================
def infer_areas(model_run="sam2_prior", sam2_ckpt=None, folds=(0, 1, 2, 3, 4),
                backbone=None, prior_mode="channel", use_prior=True):
    """Her cift icin pred_px ve gt_px (512 izgarasinda) dondurur."""
    dl = _load("dlG", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
    dl.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT)
    MM = _load("MMG", os.path.join(SCRIPTS, "03_model.py"))

    if backbone is None:
        backbone = "sam2" if model_run.startswith("sam2") else "eva02"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for k in folds:
        ck = os.path.join(RUNS_DIR, model_run, f"fold{k}_best.pt")
        if not os.path.exists(ck):
            print(f"  [atla] fold {k}: {ck} yok"); continue

        m = MM.build_model(backbone, img_size=512, pretrained=False,
                           sam2_ckpt=sam2_ckpt, prior_mode=prior_mode,
                           use_stem=True).to(dev).eval()
        sd = torch.load(ck, map_location=dev, weights_only=False)
        r = m.load_state_dict({kk: vv.float() for kk, vv in sd.items()}, strict=True)
        assert not r.missing_keys and not r.unexpected_keys

        _, va = dl.build_loaders(k, batch_size=4, augment=False)
        with torch.no_grad():
            for b in va:
                img = b["image"].to(dev)
                pri = b["prior"].to(dev)
                if not use_prior:
                    pri = None if prior_mode == "prompt" else torch.zeros_like(pri)
                p = torch.sigmoid(m(img, pri).float()).cpu().numpy()
                g = b["mask"].numpy()
                for j in range(p.shape[0]):
                    rows.append(dict(pair_id=b["pair_id"][j], fold=k,
                                     pred_px=int((p[j, 0] > 0.5).sum()),
                                     gt_px=int((g[j, 0] > 0.5).sum())))
        print(f"  fold {k}: {len(va.dataset)} cift")
        del m; torch.cuda.empty_cache()

    return pd.DataFrame(rows)


# ====================== 2) alanlari mm^2'ye cevir ========================
def build_table(areas):
    f = pd.read_csv(os.path.join(BASE_DIR, "folds_pairs.csv"))
    d = areas.merge(
        f[["pair_id", "true_patient_id", "prev_mask", "prev_mmpx",
           "mm_per_px_512", "aralik_gun", "olcek_orani"]],
        on="pair_id", how="left")
    assert d.mm_per_px_512.notna().all(), "mm_per_px_512 eksik"

    # A_prev: NATIF cozunurlukte, kendi mm/px'i ile
    prev = []
    for p, s in zip(d.prev_mask, d.prev_mmpx):
        im = imread_u(os.path.join(BASE_DIR, p))
        prev.append(float((im > 127).sum()) * s * s)
    d["A_prev"] = prev

    s512 = d.mm_per_px_512.values ** 2
    d["A_gt"]   = d.gt_px.values   * s512
    d["A_pred"] = d.pred_px.values * s512

    # yuzde degisim
    d["pct_ref"]  = 100.0 * (d.A_gt   - d.A_prev) / d.A_prev
    d["pct_pred"] = 100.0 * (d.A_pred - d.A_prev) / d.A_prev

    # mm^2/ay buyume hizi (ek)
    ay = d.aralik_gun / 30.44
    d["rate_ref"]  = (d.A_gt   - d.A_prev) / ay
    d["rate_pred"] = (d.A_pred - d.A_prev) / ay

    bos = int((d.A_prev <= 0).sum())
    if bos:
        print(f"  [uyari] A_prev = 0 olan {bos} cift yuzde analizinden cikarildi")
    return d


# ====================== 3) Bland-Altman ==================================
def bland_altman(ref, pred, etiket, log=False):
    from scipy import stats
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    ok = np.isfinite(ref) & np.isfinite(pred)
    ref, pred = ref[ok], pred[ok]

    diff = pred - ref
    mean = (pred + ref) / 2.0
    bias, sd = diff.mean(), diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

    n = len(diff)
    se_b = sd / np.sqrt(n)
    se_l = np.sqrt(3.0) * sd / np.sqrt(n)          # LoA standart hatasi
    tcrit = stats.t.ppf(0.975, n - 1)

    # orantisal yanlilik
    sl, ic, r, p_prop, _ = stats.linregress(mean, diff)
    # normallik
    W, p_norm = stats.shapiro(diff) if n <= 5000 else (np.nan, np.nan)
    # yanliligin sifirdan farki
    t_b, p_bias = stats.ttest_1samp(diff, 0.0)

    print(f"\n--- Bland-Altman: {etiket}  (n={n}) ---")
    print(f"  yanlilik (bias)      : {bias:+.2f}  %95 GA "
          f"[{bias-tcrit*se_b:+.2f}, {bias+tcrit*se_b:+.2f}]   p={p_bias:.3g}")
    print(f"  uyum sinirlari (LoA) : [{lo:+.2f}, {hi:+.2f}]")
    print(f"     LoA %95 GA        : alt [{lo-tcrit*se_l:+.2f}, {lo+tcrit*se_l:+.2f}]  "
          f"ust [{hi-tcrit*se_l:+.2f}, {hi+tcrit*se_l:+.2f}]")
    print(f"  orantisal yanlilik   : egim {sl:+.4f}  r={r:+.3f}  p={p_prop:.3g}"
          f"   {'<<< VAR, klasik LoA gecersiz' if p_prop < 0.05 else 'yok'}")
    print(f"  farklarin normalligi : Shapiro W={W:.3f}  p={p_norm:.3g}"
          f"   {'<<< IHLAL' if p_norm < 0.05 else 'ok'}")
    if p_norm < 0.05:
        q = np.percentile(diff, [2.5, 97.5])
        print(f"  yuzdelik tabanli LoA : [{q[0]:+.2f}, {q[1]:+.2f}]  (normallik ihlalinde bunu raporlayin)")
    return dict(etiket=etiket, n=n, bias=bias, sd=sd, loa_lo=lo, loa_hi=hi,
                p_bias=p_bias, prop_slope=sl, prop_p=p_prop, shapiro_p=p_norm,
                diff=diff, mean=mean)


# ====================== 4) kategorik uyum ================================
def categorize(pct, grow=GROW_TH, shrink=SHRINK_TH):
    return np.where(pct >= grow, "buyudu",
           np.where(pct <= shrink, "kuculdu", "stabil"))


def kappa_report(d, grow=GROW_TH, shrink=SHRINK_TH, etiket=""):
    from sklearn.metrics import cohen_kappa_score, confusion_matrix
    a = categorize(d.pct_ref.values, grow, shrink)
    b = categorize(d.pct_pred.values, grow, shrink)
    lab = ["kuculdu", "stabil", "buyudu"]
    cm = confusion_matrix(a, b, labels=lab)
    k_lin = cohen_kappa_score(a, b, labels=lab, weights="linear")
    k_raw = cohen_kappa_score(a, b, labels=lab)
    acc = (a == b).mean()

    print(f"\n--- Kategorik uyum {etiket} (esik +{grow:.0f}% / {shrink:.0f}%) ---")
    print("  satir = referans, sutun = tahmin")
    print(pd.DataFrame(cm, index=lab, columns=lab).to_string())
    print(f"  dogruluk {acc:.3f}   kappa {k_raw:.3f}   agirlikli kappa {k_lin:.3f}")
    return dict(acc=acc, kappa=k_raw, kappa_w=k_lin, cm=cm)


# ====================== 5) tekrarli olcum duyarliligi ====================
def per_patient_sensitivity(d, seed=42, n_rep=200):
    """Hasta basina tek cift ornekleyerek yanlilik/LoA dagilimi."""
    rng = np.random.default_rng(seed)
    B, L, H = [], [], []
    for _ in range(n_rep):
        idx = d.groupby("true_patient_id").apply(
            lambda g: g.index[rng.integers(len(g))]).values
        s = d.loc[idx]
        diff = (s.pct_pred - s.pct_ref).values
        B.append(diff.mean()); sd = diff.std(ddof=1)
        L.append(diff.mean() - 1.96 * sd); H.append(diff.mean() + 1.96 * sd)
    print(f"\n--- Hasta basina tek cift duyarliligi ({n_rep} tekrar, "
          f"n={d.true_patient_id.nunique()} hasta) ---")
    print(f"  yanlilik : {np.mean(B):+.2f}  [{np.percentile(B,2.5):+.2f}, {np.percentile(B,97.5):+.2f}]")
    print(f"  LoA alt  : {np.mean(L):+.2f}  LoA ust: {np.mean(H):+.2f}")


# ====================== 6) grafik ========================================
def plot_ba(res, path, xlab, ylab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(res["mean"], res["diff"], s=14, alpha=0.55,
               edgecolor="none", color="#2E5496")
    for y, st, lb in [(res["bias"], "-", f"yanlilik {res['bias']:+.1f}"),
                      (res["loa_lo"], "--", f"LoA alt {res['loa_lo']:+.1f}"),
                      (res["loa_hi"], "--", f"LoA ust {res['loa_hi']:+.1f}")]:
        ax.axhline(y, ls=st, lw=1.2, color="#C00000")
        ax.text(ax.get_xlim()[1], y, " " + lb, va="center", fontsize=8, color="#C00000")
    ax.axhline(0, lw=0.8, color="0.5")
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.set_title(res["etiket"], fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=200)
    print(f"  grafik: {path}")
    plt.close(fig)


# ====================== ana akis =========================================
def run(model="sam2_prior", sam2_ckpt=None, folds=(0, 1, 2, 3, 4),
        prior_mode="channel", use_prior=True, save=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"=== {model} : cikarim ===")
    areas = infer_areas(model, sam2_ckpt, folds, prior_mode=prior_mode,
                        use_prior=use_prior)
    d = build_table(areas)
    d = d[d.A_prev > 0].reset_index(drop=True)
    print(f"\nanaliz kohortu: {len(d)} cift / {d.true_patient_id.nunique()} hasta")
    print(f"referans yuzde degisim: medyan {d.pct_ref.median():+.1f}%  "
          f"IQR [{d.pct_ref.quantile(.25):+.1f}, {d.pct_ref.quantile(.75):+.1f}]")

    r_pct = bland_altman(d.pct_ref, d.pct_pred, f"{model} - yuzde alan degisimi (%)")
    r_abs = bland_altman(d.A_gt, d.A_pred, f"{model} - mutlak alan (mm^2, EK)")

    kappa_report(d, etiket="(RECIST-esdeger)")
    print("\n--- esik taramasi (agirlikli kappa) ---")
    for g, s in [(20, -20), (30, -30), (44, -51), (50, -50)]:
        from sklearn.metrics import cohen_kappa_score
        a = categorize(d.pct_ref.values, g, s); b = categorize(d.pct_pred.values, g, s)
        print(f"  +{g:>3}% / {s:>4}%   kappa_w = "
              f"{cohen_kappa_score(a, b, weights='linear'):.3f}")

    per_patient_sensitivity(d)

    if save:
        plot_ba(r_pct, os.path.join(OUT_DIR, f"fig_ba_pct_{model}.png"),
                "ortalama yuzde degisim (%)", "tahmin - referans (%)")
        plot_ba(r_abs, os.path.join(OUT_DIR, f"fig_ba_area_{model}.png"),
                "ortalama alan (mm$^2$)", "tahmin - referans (mm$^2$)")
        p = os.path.join(OUT_DIR, f"growth_{model}.csv")
        d.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"  tablo : {p}")
    return d
