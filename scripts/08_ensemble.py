# -*- coding: utf-8 -*-
"""
08_ensemble.py
--------------
AC-CA - iki omurganin olasilik duzeyinde birlestirilmesi.

NEDEN: oracle analizi (max) 0.7778 verdi, en iyi tek modelin +0.0395 ustunde.
Ancak oracle ULASILMAZ - hangi modelin hakli oldugunu bilmek dogru cevabi
bilmeyi gerektirir. Gercek topluluk performansi olculmeli.

YONTEM: her cift icin iki modelin sigmoid ciktilarini piksel bazinda
birlestirir (ortalama / maksimum / agirlikli). Fold yapisina SADIK kalir:
her cift, o cifti dogrulama setinde goren foldun agirliklariyla degerlendirilir.
Boylece sizinti olusmaz.

EGITIM YOK. Yalnizca kayitli fold*_best.pt agirliklariyla cikarim.

Kullanim (Colab):
    import importlib.util, sys
    s = importlib.util.spec_from_file_location("E", "/content/acca/scripts/08_ensemble.py")
    E = importlib.util.module_from_spec(s); sys.modules["E"] = E; s.loader.exec_module(E)
    df = E.run(sam2_ckpt="/content/sam2.1_hiera_base_plus.pt")
"""

import os
import glob
import time

import numpy as np
import pandas as pd
import torch

RUNS_DIR   = "/content/drive/MyDrive/acca_runs"
SCRIPTS    = "/content/acca/scripts"
BASE_DIR   = "/content/acca"
CACHE_ROOT = "/content/acca/_prior_cache"
IMG_SIZE   = 512
OUT_CSV    = os.path.join(RUNS_DIR, "ensemble_pairs.csv")


def _load(name, path):
    import importlib.util, sys
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); sys.modules[name] = m
    s.loader.exec_module(m); return m


def dice_iou(pred, gt):
    p, g = pred > 0.5, gt > 0.5
    sp, sg = p.sum(), g.sum()
    if sp == 0 and sg == 0:
        return 1.0, 1.0
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return (2.0 * inter / (sp + sg) if (sp + sg) else 0.0,
            inter / union if union else 0.0)


def hd95(pred, gt):
    from scipy.ndimage import distance_transform_edt as edt, binary_erosion
    p, g = pred > 0.5, gt > 0.5
    if not p.any() or not g.any():
        return np.nan
    bp = p & ~binary_erosion(p); bg = g & ~binary_erosion(g)
    if not bp.any() or not bg.any():
        return np.nan
    d = np.concatenate([edt(~g)[bp], edt(~p)[bg]])
    return float(np.percentile(d, 95))


def _build(MM, which, sam2_ckpt, dev):
    return MM.build_model(which, img_size=IMG_SIZE, pretrained=False,
                          sam2_ckpt=sam2_ckpt if which == "sam2" else None).to(dev)


def run(run_a="sam2_prior", run_b="eva02_prior",
        which_a="sam2", which_b="eva02", sam2_ckpt=None,
        weights=(0.5, 0.5), folds=(0, 1, 2, 3, 4)):
    """
    A ve B kosularinin fold*_best.pt agirliklarini kullanarak topluluk cikarimi.
    Doner: cift bazli DataFrame (dice_a, dice_b, dice_mean, dice_max, dice_wavg).
    """
    dev = torch.device("cuda")
    dl = _load("dl", os.path.join(SCRIPTS, "temporal_pair_dataloader_v2.py"))
    MM = _load("MM", os.path.join(SCRIPTS, "03_model.py"))
    dl.configure(base_dir=BASE_DIR, cache_root=CACHE_ROOT, batch_size=2)

    print(f"[i] A = {run_a} ({which_a})   B = {run_b} ({which_b})")
    rows, t0 = [], time.time()

    for k in folds:
        pa = os.path.join(RUNS_DIR, run_a, f"fold{k}_best.pt")
        pb = os.path.join(RUNS_DIR, run_b, f"fold{k}_best.pt")
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f"  [!] fold {k}: agirlik eksik, atlaniyor")
            continue

        _, va = dl.build_loaders(k, batch_size=2, augment=False)

        ma = _build(MM, which_a, sam2_ckpt, dev)
        ma.load_state_dict({q: v.float() for q, v in
                            torch.load(pa, map_location=dev, weights_only=False).items()})
        ma.eval()
        mb = _build(MM, which_b, sam2_ckpt, dev)
        mb.load_state_dict({q: v.float() for q, v in
                            torch.load(pb, map_location=dev, weights_only=False).items()})
        mb.eval()

        with torch.no_grad():
            for b in va:
                img = b["image"].to(dev); pri = b["prior"].to(dev)
                with torch.autocast("cuda", dtype=torch.float16):
                    qa = torch.sigmoid(ma(img, pri).float()).cpu().numpy()
                    qb = torch.sigmoid(mb(img, pri).float()).cpu().numpy()
                gt = b["mask"].numpy()
                for j in range(qa.shape[0]):
                    A, B, G = qa[j, 0], qb[j, 0], gt[j, 0]
                    mean = weights[0] * A + weights[1] * B
                    mx   = np.maximum(A, B)
                    da, _ = dice_iou(A, G); db, _ = dice_iou(B, G)
                    dm, im = dice_iou(mean, G); dX, _ = dice_iou(mx, G)
                    rows.append(dict(
                        pair_id=b["pair_id"][j], fold=k,
                        dice_a=da, dice_b=db,
                        dice_mean=dm, iou_mean=im, dice_pmax=dX,
                        hd95_a=hd95(A > .5, G > .5), hd95_b=hd95(B > .5, G > .5),
                        hd95_mean=hd95(mean > .5, G > .5),
                        dice_oracle=max(da, db),
                        dice_copy=float(b["dice_copy"][j])))

        del ma, mb; torch.cuda.empty_cache()
        print(f"  fold {k} bitti  ({time.time()-t0:.0f} sn)")

    df = pd.DataFrame(rows)
    os.makedirs(RUNS_DIR, exist_ok=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    report(df, which_a, which_b)
    print(f"\n[i] kaydedildi: {OUT_CSV}")
    return df


def report(df, na="SAM2", nb="EVA-02"):
    from scipy.stats import wilcoxon
    line = "=" * 68
    print("\n" + line); print(f"TOPLULUK SONUCU  (n = {len(df)})"); print(line)

    best_single = max(df.dice_a.mean(), df.dice_b.mean())
    rows = [
        (f"{na} tek",           df.dice_a.mean()),
        (f"{nb} tek",           df.dice_b.mean()),
        ("Topluluk (ortalama)", df.dice_mean.mean()),
        ("Topluluk (piksel max)", df.dice_pmax.mean()),
        ("Oracle (ULASILMAZ)",  df.dice_oracle.mean()),
        ("Taban cizgisi",       df.dice_copy.mean()),
    ]
    for n, v in rows:
        mark = ""
        if n.startswith("Topluluk"):
            mark = f"   ({v-best_single:+.4f} en iyi tek modele gore)"
        print(f"  {n:<24} Dice {v:.4f}{mark}")

    print(f"\n  HD95: {na} {df.hd95_a.mean():.1f} | {nb} {df.hd95_b.mean():.1f} "
          f"| topluluk {df.hd95_mean.mean():.1f} px")

    print("\n--- topluluk vs en iyi tek model (eslesmis) ---")
    single = df.dice_a if df.dice_a.mean() >= df.dice_b.mean() else df.dice_b
    sname  = na if df.dice_a.mean() >= df.dice_b.mean() else nb
    d = df.dice_mean - single
    try:
        w, p = wilcoxon(df.dice_mean, single)
        print(f"  topluluk - {sname}: ortalama {d.mean():+.4f}  medyan {d.median():+.4f}")
        print(f"  toplulugun kazandigi cift: {(d > 0).sum()}/{len(df)}")
        print(f"  Wilcoxon: W={w:.0f}, p={p:.3e}")
    except Exception as e:
        print(f"  [!] {e}")

    print("\n--- fold bazinda ---")
    g = df.groupby("fold").agg(**{
        na: ("dice_a", "mean"), nb: ("dice_b", "mean"),
        "topluluk": ("dice_mean", "mean"), "oracle": ("dice_oracle", "mean")})
    print(g.round(4).to_string())
    print(f"\n  topluluk 5-fold: {g['topluluk'].mean():.4f} "
          f"+/- {g['topluluk'].std(ddof=1):.4f}")
    print(line)
    print("""  NOT: oracle ulasilmazdir (hangi modelin hakli oldugunu bilmeyi
  gerektirir) ve makalede SONUC olarak verilemez; yalnizca tamamlayicilik
  potansiyelinin ust siniri olarak, gercek topluluk sonucuyla BIRLIKTE
  raporlanabilir.""")


if __name__ == "__main__":
    run(sam2_ckpt="/content/sam2.1_hiera_base_plus.pt")
