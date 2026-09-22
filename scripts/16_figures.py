# -*- coding: utf-8 -*-
"""
16_figures.py
-------------
AC-CA - Adim 16: revizyon sekilleri (Sekil 3, 4, 5, S1a).

NEDEN
  Karma protokol karari (15_last_epoch.py):
    - kollar arasi karsilastirmalar EN IYI epoch'ta kalir,
    - nnU-Net ile ilgili her sey ESIT protokolde, yani SON epoch'ta verilir
      (nnU-Net checkpoint_final ile tahmin ediyor).
  Bu yuzden:
    Sekil 3  : ana model SON epoch agirliklariyla (iki panel de); Ensemble ve
               Oracle cikarildi (EVA-02 + prior'in son epoch ciktilari eksik).
    Sekil 4  : (a) copy-forward, nnU-Net, ana model son epoch (duz) ve en iyi
               epoch (kesikli); EVA-02 + prior ve Ensemble cikarildi.
               (b) prompt kapisi (prompt kosusunun metrics.csv 'gate' sutunu).
    Sekil 5  : (a) 4 kanal, 32x32 kontrol ve mask prompt (en iyi epoch, kollar
               arasi karsilastirma; Holm p dosyalardan hesaplanir).
               (b) ana modelin Dice-boyut iliskisi (degismedi, yeniden cizildi).
    Sekil S1a: egitim egrileri; kontrol kollari dahil.
  Sekil 1 ve 2 Visio cizimleridir; bu script onlara dokunmaz.

  Tum sayilar dosyalardan hesaplanir; sekil etiketleri ve metin icin gereken
  sayilar ekrana da yazdirilir (Bulgular 3.1 ve 3.3 bunlarla guncellenir).

Kullanim (Colab, GPU gerekmez):
    F = S.load("F16", "/content/acca/scripts/16_figures.py")
    F.all_figures()          # ya da tek tek: F.figure3(), F.figure4(), F.figure5(), F.figure_s1a()

CIKTILAR: acca_runs/_figures/  (PNG 600 dpi + PDF)
"""

import os
import sys
import glob
import importlib.util

import numpy as np
import pandas as pd

# ================================ CONFIG ==============================
BASE_DIR   = "/content/acca"
CACHE_ROOT = "/content/acca/_prior_cache"
SCRIPTS    = "/content/acca/scripts"
RUNS_DIR   = "/content/drive/MyDrive/acca_runs"
OUT_DIR    = "/content/drive/MyDrive/acca_runs/_figures"
DPI        = 600

RUN_MAIN   = "sam2_prior"
RUN_LOW    = "sam2_prior_lowres32"
RUN_BIN    = "sam2_prior_binary"
RUN_PROMPT = "sam2_prior_prompt"
RUN_NOPRI  = "sam2_noprior"
RUN_EVA    = "eva02_prior"
RUN_EVANO  = "eva02_noprior"
RUN_NNU    = "nnunet_prior"

# metrics.csv once kosunun kendi klasorunde aranir; yoksa bu klasorde
# metrics_<kosu>.csv (ya da metrics_eva02prior.csv gibi eski adlar) aranir.
METRICS_DIR = "/content/drive/MyDrive/acca_runs/_metrics"

SIZE_BINS   = [0, 200, 800, 2000, 1e9]
SIZE_LABELS = ["<200", "200-800", "800-2000", ">2000"]

# Renkler mevcut sekillere gore yaklasik secildi; gerekirse buradan ayarlayin.
COL = dict(main="#1b5e46", nnunet="#c99a6b", copy="#9e9e9e",
           four="#2e6b4f", low="#7fb89a", prompt="#b5536b",
           curve="#1f2f4f")
# ======================================================================


def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=DPI, bbox_inches="tight")
    print(f"  [i] kaydedildi: {OUT_DIR}/{name}.png / .pdf")


def _pairs(run, last=False):
    suf = "_last" if (last and run != RUN_NNU) else ""
    fs = sorted(glob.glob(os.path.join(RUNS_DIR, run, f"fold*_val_pairs{suf}.csv")))
    if not fs:
        raise FileNotFoundError(f"{run}: fold*_val_pairs{suf}.csv yok")
    out = []
    for f in fs:
        x = pd.read_csv(f, encoding="utf-8-sig")
        x["fold"] = int(os.path.basename(f).split("_")[0].replace("fold", ""))
        out.append(x)
    P = pd.concat(out, ignore_index=True)
    if len(fs) != 5:
        print(f"  [!] {run}: {len(fs)} fold dosyasi (5 bekleniyor)")
    return P


def _metrics_path(run):
    """metrics.csv yolunu bulur; bulamazsa None (cokmez)."""
    cands = [os.path.join(RUNS_DIR, run, "metrics.csv"),
             os.path.join(METRICS_DIR, f"metrics_{run}.csv"),
             os.path.join(METRICS_DIR, f"metrics_{run.replace('eva02_prior', 'eva02prior')}.csv")]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def _meta():
    dl = _load("dl16", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
    dl.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT)
    df, _, _ = dl.load_table()
    return df[["pair_id", "tgt_px"]]


def _sci(p):
    """p degeri -> mathtext (4.4x10^-5 bicimi)."""
    if p >= 0.001:
        return f"{p:.3f}"
    e = int(np.floor(np.log10(p))); m = p / 10 ** e
    return rf"{m:.1f}$\times$10$^{{{e}}}$"


# ======================================================================
# SEKIL 3: nnU-Net ile esit protokolde kuyruk karsilastirmasi
# ======================================================================
def figure3():
    plt = _plt()
    main = _pairs(RUN_MAIN, last=True)[["pair_id", "dice"]].rename(columns={"dice": "ours"})
    nnu = _pairs(RUN_NNU)[["pair_id", "dice"]].rename(columns={"dice": "nn"})
    cf = _pairs(RUN_MAIN)[["pair_id", "dice_copy"]]
    M = main.merge(nnu, on="pair_id").merge(cf, on="pair_id").merge(_meta(), on="pair_id")
    print(f"\n=== SEKIL 3 (son epoch, n = {len(M)}) ===")

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.2),
                               gridspec_kw=dict(width_ratios=[1, 1.15]))
    x = np.linspace(0, 100, len(M))
    a.plot(x, np.sort(M.dice_copy), ls=":", color=COL["copy"], lw=1.4, label="Copy-forward")
    a.plot(x, np.sort(M.nn), color=COL["nnunet"], lw=1.6, label="nnU-Net (2-ch)")
    a.plot(x, np.sort(M.ours), color=COL["main"], lw=1.8, label="SAM2 + prior (main)")
    a.axhline(0.5, color="#dddddd", lw=0.8, zorder=0)
    n_nn0, n_our0 = int((M.nn == 0).sum()), int((M.ours == 0).sum())
    a.annotate(f"failure tail:\n{n_nn0} vs {n_our0} pairs at Dice = 0",
               xy=(4, 0.01), xytext=(12, 0.2), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    a.set_xlim(0, 100); a.set_ylim(0, 1.0)
    a.set_xlabel("Pairs, sorted by Dice (percentile)"); a.set_ylabel("Dice")
    a.set_title("(a) Per-pair Dice profiles (last-epoch weights)", loc="left")
    a.legend(loc="lower right", frameon=False)

    sc = b.scatter(M.nn, M.ours, c=np.log10(M.tgt_px.clip(lower=1)), cmap="viridis",
                   s=14, alpha=0.85, linewidths=0)
    b.plot([0, 1], [0, 1], ls="--", color="#999999", lw=0.8)
    nn0 = (M.nn == 0).values
    both0 = ((M.nn == 0) & (M.ours == 0)).values
    b.annotate(f"nnU-Net = 0\n(n = {nn0.sum()}, mean ours {M.ours[nn0].mean():.2f})",
               xy=(0.0, 0.8), xytext=(0.12, 0.82), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    b.annotate(f"both = 0 (n = {both0.sum()})", xy=(0.0, 0.0), xytext=(0.3, 0.08), fontsize=8,
               arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8))
    b.set_xlim(-0.02, 1.02); b.set_ylim(-0.02, 1.02)
    b.set_xlabel("nnU-Net Dice"); b.set_ylabel("SAM2 + prior Dice (last epoch)")
    b.set_title("(b) Pairwise: main model vs. nnU-Net (last epoch)", loc="left")
    cb = fig.colorbar(sc, ax=b, fraction=0.05, pad=0.02)
    cb.set_label("log$_{10}$ lesion area (px)")
    fig.tight_layout()
    _save(fig, "Figure3_last_epoch")
    plt.close(fig)

    our0 = (M.ours == 0).values
    nums = dict(n=len(M), nn_fail=n_nn0, ours_fail=n_our0, both_fail=int(both0.sum()),
                ours_in_nn_fail=float(M.ours[nn0].mean()),
                nn_in_ours_fail=float(M.nn[our0].mean()) if our0.any() else float("nan"),
                ours_empty=None)
    print(f"  tam basarisizlik: nnU-Net {n_nn0}, ana model {n_our0}, ikisi birden {both0.sum()}")
    print(f"  nnU-Net'in basarisiz oldugu {n_nn0} ciftte ana model ortalamasi: "
          f"{nums['ours_in_nn_fail']:.3f}")
    if our0.any():
        print(f"  ana modelin basarisiz oldugu {n_our0} ciftte nnU-Net ortalamasi: "
              f"{nums['nn_in_ours_fail']:.3f}  ({both0.sum()} tanesinde nnU-Net de 0)")
    print("\n  HAZIR BASLIK (Sekil 3):")
    print(f"  Figure 3. (a) Pair-level sorted Dice profiles of the primary model, nnU-Net and the "
          f"copy-forward baseline; the primary model is evaluated with last-epoch weights, as nnU-Net "
          f"is. The left tail shows complete failures (nnU-Net, {n_nn0} pairs; primary model, "
          f"{n_our0} pairs). (b) Pair-wise comparison of the primary model and nnU-Net under the same "
          f"protocol; colour indicates the logarithm of lesion area. In the {n_nn0} pairs in which "
          f"nnU-Net failed, the primary model reached a mean Dice of {nums['ours_in_nn_fail']:.2f}, "
          f"whereas {int(both0.sum())} of the {n_our0} failures of the primary model occurred in pairs "
          f"in which nnU-Net also failed.")
    print("  -> Bulgular 3.1'deki '0.431' ve '5 of the 6' cumlesi de bu sayilarla guncellenir.")
    return nums


# ======================================================================
# SEKIL 4: (a) fold bazli performans, (b) prompt kapisi
# ======================================================================
def figure4():
    plt = _plt()
    last = _pairs(RUN_MAIN, last=True).groupby("fold").dice.mean()
    best = _pairs(RUN_MAIN).groupby("fold").dice.mean()
    nnu = _pairs(RUN_NNU).groupby("fold").dice.mean()
    cf = _pairs(RUN_MAIN).groupby("fold").dice_copy.mean()
    print("\n=== SEKIL 4 ===")

    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 3.8))
    xs = np.arange(1, 6)
    a.plot(xs, cf.values, ls=":", marker="o", color=COL["copy"], ms=4, label="Copy-forward")
    a.plot(xs, nnu.values, marker="s", color=COL["nnunet"], ms=4, label="nnU-Net (2-ch)")
    a.plot(xs, last.values, marker="D", color=COL["main"], ms=4,
           label="SAM2 + prior (main), last epoch")
    a.plot(xs, best.values, ls="--", marker="D", mfc="white", color=COL["main"], ms=4,
           alpha=0.6, label="SAM2 + prior (main), best epoch")
    a.set_xticks(xs); a.set_xticklabels([f"Fold {i}" for i in xs])
    lo = min(0.3, float(min(cf.min(), nnu.min(), last.min(), best.min())) - 0.05)
    a.set_ylabel("Mean Dice"); a.set_ylim(lo, 0.85)
    a.set_title("(a) Fold-wise performance", loc="left")
    a.legend(loc="center left", bbox_to_anchor=(0.01, 0.45), frameon=False, fontsize=7)

    mp = _metrics_path(RUN_PROMPT)
    m = None
    if mp is None:
        print(f"  [!] {RUN_PROMPT} icin metrics.csv bulunamadi "
              f"({RUNS_DIR}/{RUN_PROMPT}/ ya da {METRICS_DIR}/); panel (b) bos cizilecek")
    else:
        m = (pd.read_csv(mp).drop_duplicates(["fold", "epoch"], keep="last")
             .sort_values(["fold", "epoch"]))
    if m is not None and "gate" in m.columns:
        shades = ["#9aa7bd", "#7a8aa6", "#5a6d8f", "#3c5078", COL["curve"]]
        for k, g in m.groupby("fold"):
            b.plot(g.epoch, g.gate, color=shades[int(k) % 5], lw=1.0, label=f"Fold {int(k) + 1}")
        b.axhline(0, color="#aaaaaa", lw=0.6, ls=":")
        b.set_xlabel("Epoch"); b.set_ylabel("Gate coefficient"); b.set_xlim(0, 100)
        b.legend(loc="upper right", frameon=False, ncol=2)
        fin = m.groupby("fold").gate.last()
        print(f"  gate son deger araligi: {fin.min():.3f} ile {fin.max():.3f}")
        gate_txt = f"between {fin.min():.3f} and {fin.max():.3f}".replace("-", "−")
    else:
        gate_txt = "[gate values unavailable]"
        b.text(0.5, 0.5, "gate data not found", ha="center", transform=b.transAxes)
        if m is not None:
            print("  [!] prompt metrics.csv'de 'gate' sutunu yok")
    b.set_title("(b) Prompt-pathway gate trajectory", loc="left")
    fig.tight_layout()
    _save(fig, "Figure4")
    plt.close(fig)

    for lab, v in (("ana model, son epoch", last), ("ana model, en iyi epoch", best),
                   ("nnU-Net", nnu)):
        print(f"  {lab:<24} aralik {v.min():.3f}-{v.max():.3f}   fold SD {v.std(ddof=1):.3f}")
    print("  -> 3.3'teki 'narrow band (0.709-0.791)' ifadesi ve SD karsilastirmasi "
          "son epoch satiriyla guncellenir; Ensemble cumlesi cikar ya da en iyi epoch'a baglanir.")
    print("\n  HAZIR BASLIK (Sekil 4):")
    print(f"  Figure 4. (a) Fold-wise mean Dice of the primary model with last-epoch weights "
          f"(solid) and best-epoch weights (dashed), nnU-Net and the copy-forward baseline. "
          f"(b) Epoch-wise trajectory of the learnable gate coefficient in the prompt arm across the "
          f"five folds; the coefficient converges to {gate_txt} in all folds.")
    return dict(last=last, best=best, nnunet=nnu)


# ======================================================================
# SEKIL 5: (a) boyut katmanlari (32x32 kontrolu dahil), (b) Dice-boyut
# ======================================================================
def figure5():
    plt = _plt()
    L = _load("L12f", os.path.join(SCRIPTS, "12_lowres_prior.py"))
    W = (_pairs(RUN_MAIN)[["pair_id", "dice"]].rename(columns={"dice": "four"})
         .merge(_pairs(RUN_LOW)[["pair_id", "dice"]].rename(columns={"dice": "low"}), on="pair_id")
         .merge(_pairs(RUN_PROMPT)[["pair_id", "dice"]].rename(columns={"dice": "prompt"}), on="pair_id")
         .merge(_meta(), on="pair_id"))
    W["boyut"] = pd.cut(W.tgt_px, SIZE_BINS, labels=SIZE_LABELS)
    print(f"\n=== SEKIL 5 (en iyi epoch, n = {len(W)}) ===")

    groups = [W[W.boyut == lab] for lab in SIZE_LABELS]
    p_fp = L._holm([L._wilcoxon_p(g.four, g.prompt) for g in groups])
    p_lp = L._holm([L._wilcoxon_p(g.low, g.prompt) for g in groups])

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xs = np.arange(len(SIZE_LABELS)); w = 0.26
    bars = [("four", "4-channel", COL["four"], -w), ("low", "4-channel, 32 × 32", COL["low"], 0),
            ("prompt", "Mask prompt", COL["prompt"], w)]
    for key, lab, col, off in bars:
        ax.bar(xs + off, [g[key].mean() for g in groups], w, color=col, label=lab)

    def bracket(x0, x1, y, text):
        ax.plot([x0, x0, x1, x1], [y - 0.012, y, y, y - 0.012], color="#333333", lw=0.8)
        ax.text((x0 + x1) / 2, y + 0.008, text, ha="center", va="bottom", fontsize=7)

    for i, g in enumerate(groups):
        top = max(g.four.mean(), g.low.mean(), g.prompt.mean())
        sig_fp, sig_lp = p_fp[i] < 0.05, p_lp[i] < 0.05
        if sig_fp or sig_lp:
            bracket(i, i + w, top + 0.03,
                    r"$p_{\mathrm{Holm}}$ = " + (_sci(p_lp[i]) if sig_lp else "n.s."))
            bracket(i - w, i + w, top + 0.10,
                    r"$p_{\mathrm{Holm}}$ = " + (_sci(p_fp[i]) if sig_fp else "n.s."))
        else:
            ax.text(i, top + 0.03, "n.s.", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{lab}\n(n={len(g)})" for lab, g in zip(SIZE_LABELS, groups)])
    ax.set_xlabel("Lesion area stratum (px)"); ax.set_ylabel("Mean Dice")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Injection pathway by lesion size", loc="left")
    ax.legend(loc="upper center", frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.2))

    # (b) ana model (en iyi epoch): cift duzeyi Dice - lezyon alani
    Z = W.sort_values("tgt_px").reset_index(drop=True)
    bx.axvspan(200, 800, color="#f3e6c8", alpha=0.6, lw=0)
    for a_px, lab in ((256, "1 cell"), (512, "2 cells")):
        bx.axvline(a_px, ls="--", color="#777777", lw=0.8)
        bx.text(a_px, 0.02, lab, rotation=0, ha="center", va="bottom", fontsize=7,
                color="#555555")
    bx.scatter(Z.tgt_px, Z.four, s=8, color=COL["four"], alpha=0.35, linewidths=0)
    med = Z.four.rolling(31, center=True, min_periods=8).median()
    bx.plot(Z.tgt_px, med, color=COL["main"], lw=1.6, label="Rolling median (k=31)")
    bx.set_xscale("log"); bx.set_ylim(-0.02, 1.02)
    bx.set_xlabel("Lesion area (px, log scale)"); bx.set_ylabel("Dice (4-channel model)")
    bx.set_title("(b) Main-model Dice vs. lesion size", loc="left")
    bx.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    _save(fig, "Figure5")
    plt.close(fig)

    for i, lab in enumerate(SIZE_LABELS):
        g = groups[i]
        print(f"  {lab:<9} 4ch {g.four.mean():.3f}  32x32 {g.low.mean():.3f}  prompt "
              f"{g.prompt.mean():.3f}   Holm p: 4ch-prompt {p_fp[i]:.2g}, 32x32-prompt {p_lp[i]:.2g}")
    print("  [kontrol] 4ch-prompt Holm p makale Tablo 4 ile, 32x32-prompt Tablo 5 ile ayni olmali.")
    sig = [SIZE_LABELS[i] for i in range(len(SIZE_LABELS)) if p_fp[i] < 0.05 or p_lp[i] < 0.05]
    if sig:
        band = " and ".join(x.replace("-", "–") for x in sig)
        sig_txt = (f"the differences from the prompt pathway are significant only in the {band}-pixel "
                   f"band, where the control matches the four-channel pathway")
    else:
        sig_txt = "no stratum shows a significant difference from the prompt pathway"
    print("\n  HAZIR BASLIK (Sekil 5):")
    print(f"  Figure 5. (a) Mean Dice of the four-channel pathway, the resolution control (prior "
          f"reduced to 32 × 32 before the fourth channel) and the mask-prompt pathway by size "
          f"stratum; {sig_txt}. (b) Pair-level Dice–size "
          f"relationship of the primary model (running median, k = 31); the shaded band indicates the "
          f"200–800-pixel range, and the dashed lines indicate the areas corresponding to 1 and 2 cells "
          f"on the stride-16 grid.")
    return W


# ======================================================================
# SEKIL S1a: egitim egrileri
# ======================================================================
def figure_s1a(runs=None):
    plt = _plt()
    runs = runs or [(RUN_MAIN, "SAM2 + prior"), (RUN_LOW, "SAM2 + prior, 32 × 32"),
                    (RUN_BIN, "SAM2 + prior, binary"), (RUN_PROMPT, "SAM2 prompt"),
                    (RUN_NOPRI, "SAM2 no prior"), (RUN_EVA, "EVA-02 + prior"),
                    (RUN_EVANO, "EVA-02 no prior")]
    avail = [(r, t) for r, t in runs if _metrics_path(r) is not None]
    miss = [r for r, _ in runs if (r, _) not in avail]
    if miss:
        print(f"  [!] metrics.csv yok: {miss}")
    print(f"\n=== SEKIL S1a ({len(avail)} kol) ===")
    fig, axes = plt.subplots(1, len(avail), figsize=(2.05 * len(avail), 2.5), sharey=True)
    axes = np.atleast_1d(axes)
    shades = ["#9aa7bd", "#7a8aa6", "#5a6d8f", "#3c5078", COL["curve"]]
    for ax, (run, title) in zip(axes, avail):
        m = (pd.read_csv(_metrics_path(run))
             .drop_duplicates(["fold", "epoch"], keep="last").sort_values(["fold", "epoch"]))
        for k, g in m.groupby("fold"):
            ax.plot(g.epoch, g.val_dice, color=shades[int(k) % 5], lw=0.8)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Epoch"); ax.set_xlim(0, 100); ax.set_ylim(0.25, 0.85)
    axes[0].set_ylabel("Validation Dice")
    fig.tight_layout()
    _save(fig, "FigureS1a_curves")
    plt.close(fig)
    print("\n  HAZIR BASLIK (Sekil S1a, yalnizca (a) kismi):")
    print(f"  (a) Validation Dice curves over 100 training epochs for the {len(avail)} trained SAM2 and "
          f"EVA-02 arms, including the two control arms, shown separately for each cross-validation "
          f"fold.")


def all_figures():
    out = dict(fig3=figure3(), fig4=figure4())
    figure5()
    figure_s1a()
    print("\nTumu bitti. Sekil 1 ve 2 (Visio) elle guncellenecek.")
    return out


if __name__ == "__main__":
    all_figures()
