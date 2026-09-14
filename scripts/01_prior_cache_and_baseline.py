# -*- coding: utf-8 -*-
"""
01_prior_cache_and_baseline.py
------------------------------
AC-CA calismasi, modelleme asamasi - Adim 1.

TEK GECISTE IKI IS:
  (A) PRIOR ONBELLEGI. 244 cift icin hizalanmis soft prior'i bir kez uretip
      diske yazar. Egitimde prior artik her epoch yeniden hesaplanmaz;
      dataloader sadece PNG okur. distanceTransform + 2x Otsu/kontur
      maliyeti epoch basina degil, toplamda bir kez odenir.

  (B) TABAN CIZGISI. Ayni affine gecisi hizalanmis IKILI t-1 maskesini de
      verir. "t-1 maskesini oldugu gibi tahmin say" taban cizgisinin Dice/IoU
      degeri bu maskeden bedava cikar. Modelin gecmesi gereken sayi budur.

CIKTILAR (BASE_DIR altina):
  _prior_cache/<config_id>/prior/<pair_id>.png    soft prior, uint8 [0,255]
  _prior_cache/<config_id>/pbin/<pair_id>.png     hizalanmis ikili t-1, {0,255}
  _prior_cache/<config_id>/manifest.json          config parmak izi
  _prior_cache/<config_id>/pair_stats.csv         cift basina tum metrikler

pair_id = "<patient_key>__<t_prev>_<t_curr>"  (folds_pairs.csv ile birebir)

ONEMLI: onbellek klasoru config'ten turetilen bir kimlikle isimlendirilir.
TAU_MM / DILATE_MM / ALIGN_BODY / IMG_SIZE / PRIOR_MODE degistirirseniz yeni
bir klasor olusur; eski onbellek sessizce kullanilmaz. Bayat onbellek riski yok.

Turkce yol notu: hicbir yerde cv2.imread/cv2.imwrite YOK.
Okuma  = np.fromfile + cv2.imdecode
Yazma  = cv2.imencode + ndarray.tofile

Spyder: CONFIG blogunu duzenleyip dosyayi calistirin.
"""

import os
import io
import json
import time
import hashlib
import warnings

import numpy as np
import cv2
import pandas as pd

# =============================== CONFIG ===============================
BASE_DIR   = r"C:\Users\Fatma Murat\OneDrive\Masaüstü\AC-CA Mask\Dataset_anonim"
CSV_PATH   = os.path.join(BASE_DIR, "folds_pairs.csv")
CACHE_ROOT = os.path.join(BASE_DIR, "_prior_cache")

IMG_SIZE   = 512          # onbellek ve egitim cozunurlugu

# --- prior uretimi (temporal_pair_dataloader.py K2 yapilandirmasi) ---
PRIOR_MODE = "distance"   # "distance" -> exp(-d/TAU_MM) | "blur" -> dilate+Gauss
TAU_MM     = 15.0
DILATE_MM  = 10.0         # yalnizca mode="blur" icin
SIGMA_MM   = 10.0         # yalnizca mode="blur" icin
DFOV_SKIP  = 1.10         # mm-px orani <= bu ise olcekleme atlanir
ALIGN_BODY = True         # prev->curr govde merkezi otelemesi

DEFAULT_MMPX = 0.97656

OVERWRITE  = False        # True -> mevcut onbellegi yeniden uretir
# =====================================================================


# ----------------------------- Okuma / yazma -------------------------
def robust_read_gray(path):
    buf = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Goruntu okunamadi: {path}")
    return img


def robust_read_mask(path):
    buf = np.fromfile(path, dtype=np.uint8)
    m = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if m is None:
        raise IOError(f"Maske okunamadi: {path}")
    if m.ndim == 3:
        m = m.max(axis=2)
    return (m > 0).astype(np.uint8)


def robust_write_png(path, arr):
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise IOError(f"PNG kodlanamadi: {path}")
    buf.tofile(path)


# ----------------------------- Yardimcilar ---------------------------
def body_center(img):
    """Otsu + en buyuk kontur agirlik merkezi; basarisizsa goruntu merkezi."""
    try:
        g = img if img.ndim == 2 else img[..., 0]
        _, th = cv2.threshold(g.astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            M = cv2.moments(c)
            if M["m00"] > 0:
                return (M["m10"] / M["m00"], M["m01"] / M["m00"])
    except Exception:
        pass
    H, W = img.shape[:2]
    return (W / 2.0, H / 2.0)


def align_prev_mask(prev_mask, mmpx_prev, mmpx_curr, anchor, prev_anchor):
    """
    ADIM 1+2: t-1 ikili maskesini t uzayina tasir (olcek + oteleme).
    Prior uretiminin ve taban cizgisinin ORTAK adimi.

    Doner: (aligned_binary uint8 {0,1}, s, tx, ty)
    """
    m = (prev_mask > 0).astype(np.float32)
    H, W = m.shape
    if m.sum() == 0:
        return m.astype(np.uint8), 1.0, 0.0, 0.0

    s = 1.0
    if mmpx_prev and mmpx_curr:
        ratio = max(mmpx_prev, mmpx_curr) / min(mmpx_prev, mmpx_curr)
        if ratio > DFOV_SKIP:
            s = float(mmpx_prev) / float(mmpx_curr)

    tx = ty = 0.0
    if ALIGN_BODY and prev_anchor is not None:
        tx = float(anchor[0]) - float(prev_anchor[0])
        ty = float(anchor[1]) - float(prev_anchor[1])

    if s != 1.0 or tx != 0.0 or ty != 0.0:
        cx, cy = anchor
        M = np.array([[s, 0, cx * (1 - s) + tx],
                      [0, s, cy * (1 - s) + ty]], dtype=np.float32)
        m = cv2.warpAffine(m, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        m = (m > 0.5).astype(np.float32)

    return m.astype(np.uint8), s, tx, ty


def encode_prior(aligned_bin, mmpx_curr):
    """ADIM 3+4: hizalanmis ikili maskeyi soft prior'a cevirir. [0,1] float32."""
    m = (aligned_bin > 0).astype(np.float32)
    if m.sum() == 0:
        return m

    if PRIOR_MODE == "distance":
        inv = (m < 0.5).astype(np.uint8)
        d_px = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        d_mm = d_px * float(mmpx_curr)
        m = np.exp(-d_mm / float(TAU_MM)).astype(np.float32)
    else:
        dil_px = max(1, int(round(DILATE_MM / mmpx_curr)))
        k = 2 * dil_px + 1
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        sig_px = max(0.5, SIGMA_MM / mmpx_curr)
        ksz = max(3, int(2 * round(3 * sig_px) + 1))
        m = cv2.GaussianBlur(m, (ksz, ksz), sigmaX=sig_px, sigmaY=sig_px)

    mx = m.max()
    if mx > 0:
        m = m / mx
    return m.astype(np.float32)


def dice_iou(a, b):
    """a, b: ikili diziler. Ikisi de bossa (1,1); biri bossa (0,0)."""
    a = (a > 0); b = (b > 0)
    sa, sb = a.sum(), b.sum()
    if sa == 0 and sb == 0:
        return 1.0, 1.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    d = 2.0 * inter / (sa + sb) if (sa + sb) > 0 else 0.0
    j = inter / union if union > 0 else 0.0
    return float(d), float(j)


def centroid(mask):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        return None
    return (xs.mean(), ys.mean())


def config_id():
    cfg = dict(mode=PRIOR_MODE, tau=TAU_MM, dil=DILATE_MM, sig=SIGMA_MM,
               skip=DFOV_SKIP, align=ALIGN_BODY, size=IMG_SIZE)
    h = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:6]
    tag = f"{PRIOR_MODE}_tau{TAU_MM:g}" if PRIOR_MODE == "distance" \
          else f"blur_d{DILATE_MM:g}_s{SIGMA_MM:g}"
    return f"{tag}_align{int(ALIGN_BODY)}_sz{IMG_SIZE}_{h}", cfg


# ------------------------------- Ana gecis ---------------------------
def build(csv_path=CSV_PATH, base_dir=BASE_DIR, cache_root=CACHE_ROOT):
    cid, cfg = config_id()
    cache_dir = os.path.join(cache_root, cid)
    prior_dir = os.path.join(cache_dir, "prior")
    pbin_dir  = os.path.join(cache_dir, "pbin")
    for d in (prior_dir, pbin_dir):
        os.makedirs(d, exist_ok=True)

    print(f"[i] Onbellek kimligi : {cid}")
    print(f"[i] Onbellek klasoru : {cache_dir}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    need = {"patient_key", "t_prev", "t_curr", "prev_img", "curr_img",
            "prev_mask", "curr_mask", "prev_mmpx", "curr_mmpx", "fold"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"folds_pairs.csv'de eksik sutun: {sorted(missing)}")

    rows = []
    t0 = time.time()
    for i, r in df.iterrows():
        pair_id = f"{r['patient_key']}__{r['t_prev']}_{r['t_curr']}"

        curr_img = robust_read_gray(os.path.join(base_dir, str(r["curr_img"])))
        Hc, Wc = curr_img.shape
        prev_img = robust_read_gray(os.path.join(base_dir, str(r["prev_img"])))
        if prev_img.shape != (Hc, Wc):
            prev_img = cv2.resize(prev_img, (Wc, Hc), interpolation=cv2.INTER_AREA)

        prev_m = robust_read_mask(os.path.join(base_dir, str(r["prev_mask"])))
        if prev_m.shape != (Hc, Wc):
            prev_m = cv2.resize(prev_m, (Wc, Hc), interpolation=cv2.INTER_NEAREST)
        curr_m = robust_read_mask(os.path.join(base_dir, str(r["curr_mask"])))
        if curr_m.shape != (Hc, Wc):
            curr_m = cv2.resize(curr_m, (Wc, Hc), interpolation=cv2.INTER_NEAREST)

        mmpx_prev = float(r["prev_mmpx"]) if pd.notna(r["prev_mmpx"]) else None
        mmpx_curr = float(r["curr_mmpx"]) if pd.notna(r["curr_mmpx"]) else DEFAULT_MMPX

        anchor = body_center(curr_img)
        prev_anchor = body_center(prev_img) if ALIGN_BODY else None

        pbin, s, tx, ty = align_prev_mask(prev_m, mmpx_prev, mmpx_curr,
                                          anchor, prev_anchor)
        prior = encode_prior(pbin, mmpx_curr)

        # --- egitim cozunurlugune indir ---
        S = IMG_SIZE
        prior_s = cv2.resize(prior, (S, S), interpolation=cv2.INTER_LINEAR)
        pbin_s  = cv2.resize(pbin,  (S, S), interpolation=cv2.INTER_NEAREST)
        praw_s  = cv2.resize(prev_m, (S, S), interpolation=cv2.INTER_NEAREST)
        curr_s  = cv2.resize(curr_m, (S, S), interpolation=cv2.INTER_NEAREST)
        # olcek degistiyse mm/px de degisir (metrikler mm cinsinden)
        mm_s = mmpx_curr * (Wc / float(S))

        # --- onbellege yaz ---
        p_png = os.path.join(prior_dir, pair_id + ".png")
        b_png = os.path.join(pbin_dir,  pair_id + ".png")
        if OVERWRITE or not os.path.exists(p_png):
            robust_write_png(p_png, np.clip(prior_s * 255.0, 0, 255).astype(np.uint8))
        if OVERWRITE or not os.path.exists(b_png):
            robust_write_png(b_png, (pbin_s * 255).astype(np.uint8))

        # ------------------------- METRIKLER -------------------------
        # (1) TABAN CIZGISI: hizalanmis t-1 maskesi = tahmin
        d_al, j_al = dice_iou(pbin_s, curr_s)
        # (2) hizalamasiz cikplak kopya (naif taban cizgisi)
        d_raw, j_raw = dice_iou(praw_s, curr_s)

        # (3) soft prior'in hedefi kapsamasi
        tgt = (curr_s > 0)
        n_tgt = int(tgt.sum())
        cov_soft = float((prior_s * tgt).sum() / n_tgt) if n_tgt else np.nan
        cov_bin  = float(np.logical_and(pbin_s > 0, tgt).sum() / n_tgt) if n_tgt else np.nan
        touch    = bool(np.logical_and(pbin_s > 0, tgt).any())

        # (4) alan ve kalinti kayma
        c_al, c_tg = centroid(pbin_s), centroid(curr_s)
        disp_mm = (np.hypot(c_al[0] - c_tg[0], c_al[1] - c_tg[1]) * mm_s
                   if (c_al and c_tg) else np.nan)
        n_prev = int((praw_s > 0).sum())

        rows.append(dict(
            pair_id=pair_id,
            patient_key=r["patient_key"], t_prev=r["t_prev"], t_curr=r["t_curr"],
            fold=int(r["fold"]), strat=r.get("strat", ""),
            aralik_gun=r.get("aralik_gun", np.nan),
            dice_copy_aligned=d_al, iou_copy_aligned=j_al,
            dice_copy_raw=d_raw,   iou_copy_raw=j_raw,
            cov_soft=cov_soft, cov_bin=cov_bin, touch=touch,
            prior_area_frac=float(prior_s.mean()),
            tgt_area_frac=float(n_tgt) / (S * S),
            tgt_px=n_tgt, prev_px=n_prev,
            area_ratio=(n_tgt / n_prev) if n_prev else np.nan,
            disp_mm=disp_mm,
            scale_s=s, shift_x=tx, shift_y=ty,
            native_h=Hc, native_w=Wc, mmpx_curr=mmpx_curr,
        ))

        if (i + 1) % 25 == 0:
            print(f"    {i+1}/{len(df)} ...")

    stats = pd.DataFrame(rows)
    stats_path = os.path.join(cache_dir, "pair_stats.csv")
    stats.to_csv(stats_path, index=False, encoding="utf-8-sig")

    with open(os.path.join(cache_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(dict(config_id=cid, config=cfg, n_pairs=len(stats),
                       csv=os.path.basename(csv_path),
                       created=time.strftime("%Y-%m-%d %H:%M:%S")),
                  f, indent=2, ensure_ascii=False)

    print(f"[i] {len(stats)} cift islendi ({time.time()-t0:.1f} sn)")
    print(f"[i] Istatistikler -> {stats_path}")
    return stats, cache_dir


# ------------------------------- Rapor -------------------------------
def report(stats):
    S = stats
    print("\n" + "=" * 68)
    print("TABAN CIZGISI: t-1 maskesini dogrudan tahmin saymak")
    print("=" * 68)

    def line(name, col):
        v = S[col].astype(float)
        print(f"  {name:<26} ort={v.mean():.4f}  medyan={v.median():.4f}  "
              f"sd={v.std():.4f}  min={v.min():.3f}  maks={v.max():.3f}")

    line("Dice (hizalanmis kopya)", "dice_copy_aligned")
    line("Dice (ham kopya)",        "dice_copy_raw")
    line("IoU  (hizalanmis kopya)", "iou_copy_aligned")
    kaz = S.dice_copy_aligned.mean() - S.dice_copy_raw.mean()
    print(f"\n  -> govde hizalamasinin Dice kazanci: {kaz:+.4f}")

    print("\n--- fold bazinda (hizalanmis kopya Dice) ---")
    g = S.groupby("fold").dice_copy_aligned.agg(["count", "mean", "median", "std"])
    print(g.round(4).to_string())
    print(f"\n  5-fold ortalamalarin ortalamasi = {g['mean'].mean():.4f} "
          f"(+/- {g['mean'].std():.4f})")

    print("\n--- takip araligina gore ---")
    bins = [-0.1, 30, 90, 365, 1e9]
    lab = ["<30 gun", "30-90", "90-365", ">365"]
    S2 = S.dropna(subset=["aralik_gun"]).copy()
    S2["arb"] = pd.cut(S2.aralik_gun.astype(float), bins=bins, labels=lab)
    print(S2.groupby("arb", observed=True)
            .agg(n=("dice_copy_aligned", "size"),
                 dice=("dice_copy_aligned", "mean"),
                 cov=("cov_soft", "mean"),
                 alan_orani=("area_ratio", "median")).round(4).to_string())
    print(f"  (tarihi olmayan {len(S)-len(S2)} cift haric)")

    print("\n--- prior kalitesi ---")
    print(f"  cov_soft medyan      : {S.cov_soft.median():.4f}")
    print(f"  cov_bin  medyan      : {S.cov_bin.median():.4f}")
    print(f"  prior alan orani ort : {S.prior_area_frac.mean()*100:.2f}% goruntunun")
    print(f"  hedef alan orani ort : {S.tgt_area_frac.mean()*100:.3f}% goruntunun")
    print(f"  prior/hedef alan kati: {S.prior_area_frac.mean()/S.tgt_area_frac.mean():.1f}x")
    nt = sorted(S.loc[~S.touch, "pair_id"].tolist())
    print(f"  prior hedefe DEGMEYEN: {len(nt)} cift "
          f"({100*len(nt)/len(S):.1f}%)  {nt[:8]}{' ...' if len(nt) > 8 else ''}")
    if len(nt) > 0.05 * len(S):
        print("  !! UYARI: degmeyen cift orani %5'i asiyor. Beklenen ~%1. "
              "Hizalama veya maske eslesmesi bozuk olabilir - qc_overlay ile bakin.")
    print(f"  kalinti kayma (mm)   : medyan={S.disp_mm.median():.1f}  "
          f"%90={S.disp_mm.quantile(0.90):.1f}  maks={S.disp_mm.max():.1f}")

    print("\n--- sinif dengesizligi (Focal-Tversky ayari icin) ---")
    print(f"  pozitif piksel orani : {S.tgt_area_frac.mean()*100:.3f}%  "
          f"-> arka plan/lezyon = {1/S.tgt_area_frac.mean():.0f}:1")
    print(f"  hedef alan (px) medyan={S.tgt_px.median():.0f}  "
          f"min={S.tgt_px.min():.0f}  maks={S.tgt_px.max():.0f}")

    print("\n--- en kotu 10 cift (modelin kazanacagi yer) ---")
    w = S.nsmallest(10, "dice_copy_aligned")[
        ["pair_id", "fold", "dice_copy_aligned", "cov_soft",
         "disp_mm", "area_ratio", "aralik_gun"]]
    print(w.round(3).to_string(index=False))
    print("=" * 68)


if __name__ == "__main__":
    stats, cache_dir = build()
    report(stats)
    print(f"\n[OK] Onbellek hazir: {cache_dir}")
    print("[>] Sonraki adim: dataloader v2 bu klasordeki prior/ PNG'lerini okuyacak.")
